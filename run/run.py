import os
import time
from functools import partial

import SimpleITK as sitk
import numpy as np
import torch
from monai import transforms
from monai.data import decollate_batch
from torch.cuda.amp import GradScaler, autocast

from evaluation.evaluator import aggregate_scores
from loaders_and_transforms.loaders_and_transforms import get_loader
from loss.loss import AverageMeter
from networks.get_model import get_model
from optimizer_and_scheduler.optimizer import poly_learning_rate
from utils.custom_sliding_window_inference import sliding_window_inference
from utils.utils import distributed_all_gather


def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_func,
                 val_metric,
                 args,
                 model_inferer=None,
                 scheduler=None,
                 post_label=None,
                 post_pred=None,
                 start_epoch=1,
                 saved_values=None,
                 device="cuda",
                 my_run=None
                 ):
    scaler = None
    if args.amp:
        scaler = GradScaler()

    generalized_dice_max = 0.

    if saved_values is not None:
        eval_criterion = saved_values["eval_criterion"]
        best_criterion = saved_values["best_criterion"]
        best_liver_dice = saved_values["best_liver_dice"]
        best_tumor_dice = saved_values["best_tumor_dice"]
        best_generalized_dice = saved_values["best_generalized_dice"]
        if args.early_stopping:
            train_criterion = saved_values["train_criterion"]
            best_train_criterion = saved_values["best_train_criterion"]
            best_epoch_based_on_train_loss = int(saved_values["best_epoch_based_on_train_loss"])

    for epoch in range(start_epoch, args.epochs + 1):
        if args.distributed:
            torch.distributed.barrier()
        if args.rank == 0:
            print(time.ctime(), 'Epoch:', epoch)
        epoch_time = time.time()
        train_loss, timer = train_epoch(model,
                                        train_loader,
                                        optimizer,
                                        scaler=scaler,
                                        epoch=epoch,
                                        loss_func=loss_func,
                                        args=args,
                                        device=device,
                                        scheduler=scheduler)

        if args.rank == 0:
            print('Final training  {}/{}'.format(epoch, args.epochs), 'loss: {:.4f}'.format(train_loss),
                  'time {:.2f}s'.format(time.time() - epoch_time),
                  'load time {:.2f}s'.format(timer[1]),
                  'time_before_calc {:.2f}s'.format(timer[2]),
                  'calc time {:.2f}s'.format(timer[3]),
                  'back time {:.2f}s'.format(timer[4]),
                  )

            my_run.log({'train loss': train_loss, 'epoch': epoch})

        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            dice_per_struc, val_loss, generalized_dice = val_epoch(model,
                                                                   val_loader,
                                                                   epoch=epoch,
                                                                   val_metric=val_metric,
                                                                   loss_func=loss_func,
                                                                   model_inferer=model_inferer,
                                                                   args=args,
                                                                   post_label=post_label,
                                                                   post_pred=post_pred,
                                                                   device=device)

            if args.rank == 0:
                if len(dice_per_struc) == 1:
                    dice_per_struc = np.insert(dice_per_struc, 0, 0)

                if epoch == 1:
                    best_liver_dice = dice_per_struc[0]
                    best_tumor_dice = dice_per_struc[1]
                    best_generalized_dice = generalized_dice

                print('Final validation  {}/{}'.format(epoch, args.epochs), 'generalized dice: ', generalized_dice,
                      'time {:.2f}s'.format(time.time() - epoch_time))

                my_run.log({'liver_acc': dice_per_struc[0],
                            'tumor_acc': dice_per_struc[1], 'val_loss': val_loss,
                            'generalized_dice_val': generalized_dice,
                            'epoch': epoch,
                            'best_liver_dice': max(best_liver_dice, dice_per_struc[0]),
                            'best_tumor_dice': max(best_tumor_dice, dice_per_struc[1]),
                            'best_generalized_dice_val': max(best_generalized_dice, generalized_dice)
                            })

            if args.use_eval_criterion:
                if epoch == 1:
                    eval_criterion = generalized_dice
                    best_criterion = eval_criterion
                else:
                    eval_criterion = eval_criterion * args.eval_criterion_factor + (
                            1 - args.eval_criterion_factor) * generalized_dice
                    if args.rank == 0:
                        my_run.log({'eval_criterion': eval_criterion, 'epoch': epoch})
            else:
                eval_criterion = generalized_dice
                best_criterion = generalized_dice_max

        if args.early_stopping:
            if epoch == 1:
                train_criterion = train_loss
                best_train_criterion = train_criterion
                best_epoch_based_on_train_loss = epoch
            else:
                train_criterion = train_criterion * args.train_criterion_factor + (
                        1 - args.train_criterion_factor) * train_loss
            if args.rank == 0:
                my_run.log({'train_criterion': train_criterion, 'epoch': epoch})

            if train_criterion + args.train_criterion_eps < best_train_criterion:
                best_train_criterion = train_criterion
                best_epoch_based_on_train_loss = epoch

            if epoch - best_epoch_based_on_train_loss > args.nb_epoch_for_patience:
                if optimizer.param_groups[0]['lr'] > args.lr_threshold:
                    best_epoch_based_on_train_loss = epoch - args.nb_epoch_for_patience // 2

                else:
                    if args.rank == 0:
                        print("The training loss isn't progressing anymore since: " + str(args.nb_epoch_for_patience) +
                              "epochs and the learning rate is low enough. Early stopping at epoch: " + str(epoch))
                    break

        if args.poly_learning_rate:
            poly_learning_rate(optimizer, epoch + 1, args.epochs, args.optim_lr, exponent=0.9)

        if ((epoch + 1) % args.val_every == 0 or epoch == 1) and args.rank == 0:
            saved_values = {
                "eval_criterion": eval_criterion,
                "best_criterion": max(eval_criterion, best_criterion),
                "best_liver_dice": max(dice_per_struc[0], best_liver_dice),
                "best_tumor_dice": max(dice_per_struc[1], best_tumor_dice),
                "best_generalized_dice": max(generalized_dice, best_generalized_dice)
            }
            if args.early_stopping:
                saved_values = {
                    "best_train_criterion": best_train_criterion,
                    "best_epoch_based_on_train_loss": best_epoch_based_on_train_loss,
                    "train_criterion": train_criterion
                }

            if eval_criterion > best_criterion or epoch == 1:
                print('new best criterion ({:.6f} --> {:.6f}). '.format(best_criterion, eval_criterion))
                best_criterion = eval_criterion
                if args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(model, epoch, args,
                                    saved_values=saved_values,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    filename='best_model.pt')

            if dice_per_struc[0] > best_liver_dice or epoch == 1:
                print('new best liver dice ({:.6f} --> {:.6f}). '.format(best_liver_dice, dice_per_struc[0]))
                best_liver_dice = dice_per_struc[0]
                if args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(model, epoch, args,
                                    saved_values=saved_values,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    filename='best_liver_model.pt')

            if dice_per_struc[1] > best_tumor_dice or epoch == 1:
                print('new best tumor dice ({:.6f} --> {:.6f}). '.format(best_tumor_dice, dice_per_struc[1]))
                best_tumor_dice = dice_per_struc[1]
                if args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(model, epoch, args,
                                    saved_values=saved_values,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    filename='best_tumor_model.pt')

            if generalized_dice > best_generalized_dice or epoch == 1:
                print('new best generalized dice ({:.6f} --> {:.6f}). '.format(best_generalized_dice, generalized_dice))
                best_liver_dice = generalized_dice
                if args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(model, epoch, args,
                                    saved_values=saved_values,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    filename='best_generalized_model.pt')

            save_checkpoint(model, epoch, args,
                            saved_values=saved_values,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            filename='last_epoch_model.pt')

            my_run.log({'best_eval_criterion': best_criterion, 'epoch': epoch})

    print('Training Finished !, Best criterion: ', best_criterion)

    if args.save_val_pred and args.rank == 0:
        best_model = get_model(args)
        best_model_dict = torch.load(os.path.join(args.logdir, 'best_model.pt'))
        best_model.load_state_dict(best_model_dict['state_dict'])

        args.pad_val = False
        args.crop_val = False
        args.distributed = False
        _, val_loader = get_loader(args)
        args.distributed = True
        save_prediction_to_nifty(best_model, val_loader, args, device)

    return best_criterion


def train_epoch(model,
                loader,
                optimizer,
                scaler,
                epoch,
                loss_func,
                args,
                device='cuda',
                scheduler=None
                ):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    timer = [0, 0, 0, 0, 0]

    for idx, batch_data in enumerate(loader):
        load_time = time.time()
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']

        if args.do_ds:
            data = data.to(device, non_blocking=True)
            for i, tensor in enumerate(target):
                target[i] = tensor.to(device, non_blocking=True)

        else:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        for param in model.parameters():
            param.grad = None
        time_before_calc = time.time()
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)

        calc_time = time.time()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            if args.batch_scheduler and scheduler is not None:
                scheduler.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True
                                               )
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.nb_node)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        back_time = time.time()

        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.epochs, idx + 1, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time),
                  'load time {:.2f}s'.format(load_time - start_time),
                  'time_before_calc {:.2f}s'.format(time_before_calc - load_time),
                  'calc time {:.2f}s'.format(calc_time - time_before_calc),
                  'back time {:.2f}s'.format(back_time - calc_time),
                  )
            if np.isnan(run_loss.avg):
                print("the value of the loss is NaN, interrupting code.")
                raise SystemExit
        timer[0] += time.time() - start_time
        timer[1] += load_time - start_time
        timer[2] += time_before_calc - load_time
        timer[3] += calc_time - time_before_calc
        timer[4] += back_time - calc_time

        start_time = time.time()

    for param in model.parameters():
        param.grad = None
    return run_loss.avg, timer


def val_epoch(model,
              loader,
              epoch,
              val_metric,
              loss_func,
              args,
              model_inferer=None,
              post_label=None,
              post_pred=None,
              device='cuda'):
    model.to(device)
    model.eval()
    start_time = time.time()
    # that list allow to store the computed metric per structure studied e.g. with liver and tumour we will have
    # two sublist with the patch_size of the validation set
    per_struc_list = [[] for _ in range(
        args.out_channels - 1)]
    dice_per_struc = np.zeros(args.out_channels - 1, float)
    val_loss = AverageMeter()
    generalized_dice = []
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data['image'], batch_data['label']

            if args.do_ds:
                data = data.to(device)
                for i, tensor in enumerate(target):
                    target[i] = tensor.to(device, non_blocking=True)
            else:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)

                loss = loss_func(logits, target)

                if args.distributed:
                    loss_list = distributed_all_gather([loss],
                                                       out_numpy=True,
                                                       is_valid=idx < loader.sampler.valid_length
                                                       )
                    val_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                                    n=1 * args.nb_node)
                else:
                    val_loss.update(loss.item(), n=1)

            if args.do_ds:
                target = target[0]
                logits = logits[0]

            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            val_metric.reset()
            metric = val_metric(val_output_convert, val_labels_convert)

            if args.distributed:
                generalized_dice += distributed_all_gather(metric[0], out_numpy=True,
                                                           is_valid=idx < loader.sampler.valid_length
                                                           )[0]
            else:
                generalized_dice.append(metric[0].item())
            metric = metric[1]

            metric = metric.to(device)

            if args.distributed:
                metric_list = distributed_all_gather([metric[0]], out_numpy=True,
                                                     is_valid=idx < loader.sampler.valid_length
                                                     )[0]
            else:
                metric_list = metric.detach().cpu().numpy()

            for patient in metric_list:
                for j, struc in enumerate(per_struc_list):
                    struc.append(patient[j])

            avg_metric = np.nanmean(metric_list)
            if args.rank == 0:
                print('Val {}/{} {}/{}'.format(epoch, args.epochs, idx + 1, len(loader)),
                      'average dice', avg_metric,
                      'time {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()

    for i, struc in enumerate(per_struc_list):
        dice_per_struc[i] = np.nanmean(struc)

    generalized_dice = np.nanmean(generalized_dice)

    return dice_per_struc, val_loss.avg, generalized_dice


def save_prediction_to_nifty(model, loader, args, device, test=False):
    print('Saving segmentations to nifty files')
    model.eval()
    model.to(device)
    model_inferer = partial(sliding_window_inference,
                            roi_size=[args.roi_x, args.roi_y, args.roi_z],
                            scales=torch.tensor(
                                args.deep_supervision_scales[:args.ds_retained_len]) if args.do_ds else None,
                            sw_batch_size=args.batch_size,
                            predictor=model,
                            overlap=args.infer_overlap)

    path = os.path.join(args.logdir, 'validation_raw')
    if test:
        path = os.path.join(args.logdir, 'test_raw')
    if not os.path.isdir(path):
        os.mkdir(path)
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data['image'], batch_data['label']

            if args.do_ds:
                data = data.to(device)
                for i, tensor in enumerate(target):
                    target[i] = tensor.to(device, non_blocking=True)
            else:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            with autocast(enabled=args.amp):

                segmentation = model_inferer(data)
                segmentation = segmentation[0]
                if args.do_ds:
                    segmentation = segmentation[0]
                if args.use_sigmoid or args.use_softmax:
                    if args.use_sigmoid:
                        act = transforms.Activations(sigmoid=args.use_sigmoid)
                    else:
                        act = transforms.Activations(softmax=args.use_softmax)
                    segmentation = act(segmentation)
                disc = transforms.AsDiscrete(argmax=True
                                             )

                segmentation = disc(segmentation)
                ref_seg_img = sitk.ReadImage(batch_data['label_meta_dict']['filename_or_obj'][0])

                dim = (ref_seg_img.GetWidth(), ref_seg_img.GetHeight(), ref_seg_img.GetDepth())
                pad = transforms.SpatialPad(dim)
                crop = transforms.CenterSpatialCrop(dim)
                flip_x = transforms.Flip(spatial_axis=0)
                flip_y = transforms.Flip(spatial_axis=1)
                # We need to pad AND crop to the image dimensions to be sure to have the good dimensions trust me
                segmentation = pad(
                    segmentation)
                segmentation = crop(segmentation)
                segmentation = flip_x(segmentation)
                segmentation = flip_y(segmentation)
                segmentation = torch.transpose(segmentation.squeeze(0), 0, 2)
                segmentation = segmentation.cpu().numpy()
                segmentation = sitk.GetImageFromArray(segmentation)
                segmentation.CopyInformation(ref_seg_img)
                if args.save_to_original_spacing:  # if you want to upsample / downsample your images to the original
                    # spacing of your images put the non resampled labels into a folder named "non_resampled_labels"
                    ref_seg_img_not_resampled = sitk.ReadImage(os.path.join(args.data_dir, "non_resampled_labels",
                                                                            batch_data['label_meta_dict'][
                                                                                'filename_or_obj'][0].split('/')[-1]))
                    resample = sitk.ResampleImageFilter()
                    resample.SetInterpolator(sitk.sitkNearestNeighbor)
                    resample.SetOutputDirection(ref_seg_img_not_resampled.GetDirection())
                    resample.SetOutputOrigin(ref_seg_img_not_resampled.GetOrigin())
                    resample.SetOutputSpacing(ref_seg_img_not_resampled.GetSpacing())
                    resample.SetSize(ref_seg_img_not_resampled.GetSize())
                    segmentation = resample.Execute(segmentation)

                patient_id = batch_data['label_meta_dict']['filename_or_obj'][0].split('/')[-1]
                sitk.WriteImage(segmentation, f"{path}/im{patient_id[2:]}")

    if not os.path.isdir(path):
        os.mkdir(path)
    pred_gt_tuples = []
    for i, p in enumerate(os.listdir(path)):
        if p.endswith('nii.gz'):
            file = os.path.join(path, p)
            if args.save_to_original_spacing:
                pred_gt_tuples.append([file, os.path.join(args.data_dir, 'non_resampled_labels', 'lb' + p[2:])])
            else:
                pred_gt_tuples.append([file, os.path.join(args.data_dir, 'labelsTr', 'lb' + p[2:])])

    _ = aggregate_scores(pred_gt_tuples, labels=args.classes,
                         json_output_file=os.path.join(path, "summary.json"), num_threads=8,
                         get_tumor_burden=args.get_tumor_burden)

    print("Predictions have been saved in: " + path)


def save_checkpoint(model,
                    epoch,
                    args,
                    filename='model.pt',
                    saved_values=None,
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
        'epoch': epoch,
        'state_dict': state_dict,
        'saved_values': saved_values
    }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)
