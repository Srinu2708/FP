# library import
import argparse
import os
from collections import OrderedDict
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import wandb
from monai.transforms import AsDiscrete, Activations, Compose
from monai.utils.enums import MetricReduction
from torch.nn.parallel import DistributedDataParallel as DDP

from config.config import get_config
from idr_torch_variables.idr_torch import initialize_rank
from loaders_and_transforms.loaders_and_transforms import get_loader
from loss.loss import get_loss
from loss.metric import GeneralizedDiceScore
from networks.get_model import get_model
from optimizer_and_scheduler.optimizer import get_optimizer, poly_learning_rate, optimizer_to_device
from optimizer_and_scheduler.scheduler import get_scheduler
from run.run import run_training
from utils.custom_sliding_window_inference import sliding_window_inference

parser = argparse.ArgumentParser(description='Segmentation pipeline')

parser.add_argument('--config_file', default='./config/test/transbts_config.yaml', type=str,
                    help='configuration file to be used')
parser.add_argument('--debug', default=False, type=bool, help='Use non threaded dataloader for an easier debugging')

# Model
parser.add_argument('--model_name', default='model_name', type=str, help='model name to be used for training')
parser.add_argument('--id', default='optimised', type=str, help='run name')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--pretrained_weight', default=None, type=str, help='use pretrained weight')

# Training parameters
parser.add_argument('--epochs', default=1000, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=1, type=int, help='number elements by batch')
parser.add_argument('--val_every', default=1, type=int, help='validation frequency')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--fold_id', default=0, type=int, help='fold id to validate on (between 0 and 4)')
parser.add_argument('--kfold', default=True, type=bool, help='use fold or direct validation')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--seed', default=1, type=int, help='random seed used for fold split')
parser.add_argument('--amp', default=True, help='use amp for training')

# Distributed training
parser.add_argument('--distributed', default=False, type=bool, help='multi gpu or not')
parser.add_argument('--nb_node', default=1, type=int, help='multi gpu or not')

# Loss
parser.add_argument('--loss', default="GeneralizedDiceLoss", type=str, help='loss function used for training')
parser.add_argument('--do_ds', default=False, type=bool, help='use deep supervision or not')
parser.add_argument('--metric', default="GeneralizedDice", type=str, help='loss function used for training')
parser.add_argument('--loss_reduction', default="mean", type=str, help='loss function used for training')
parser.add_argument('--use_softmax', default=True, type=bool, help='use softmax before loss calculation')
parser.add_argument('--use_sigmoid', default=False, type=bool, help='use sigmoid before loss calculation')
parser.add_argument('--include_background', default=True, type=bool,
                    help='include background during loss and metrics calculations')
parser.add_argument('--dice_prop', default=0.5, type=float,
                    help='proportion of dice compared to cross entropy for dice and ce loss')
parser.add_argument('--smooth_dr', default=1e-6, type=float, help='constant added to dice denominator to avoid nan')
parser.add_argument('--smooth_nr', default=0.0, type=float, help='constant added to dice numerator to avoid zero')

# Optimizer and scheduler
parser.add_argument('--optimizer', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--poly_learning_rate', default=False, type=bool,
                    help='update learning rate based on nnunet poly learning rate implementation')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--amsgrad', default=False, type=int, help='use amsgrad for optimizer')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=0, type=int, help='number of warmup epochs')

# Data location
parser.add_argument('--data_dir', default='../db_delineate_sota/', type=str, help='dataset directory')
parser.add_argument('--personalized_dir', default=True, type=bool,
                    help='store the output in a directory with model name and date')
parser.add_argument('--id_dir', default=False, type=bool, help='replace the name of the model by its id')
parser.add_argument('--json_list', default='dataset.json', type=str, help='dataset json file')
parser.add_argument('--save_checkpoint', default=True, help='save checkpoint during training')

# Model architecture
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=3, type=int, help='number of classes')
parser.add_argument('--classes', default=[[1, 2], 2], type=int, help='classes used to perform inference,'
                                                                     ' here whole liver (healthy liver + tumor)'
                                                                     ' and tumor only')

# UNETR
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')
parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads in ViT encoder')
parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp dimension in ViT encoder')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden patch_size dimension in ViT encoder')
parser.add_argument('--feature_size', default=32, type=int, help='feature patch_size dimension')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')

# SWIN-UNETR
parser.add_argument('--dropout_path_rate', default=0.0, type=float, help='drop path rate')

# VT-UNET
parser.add_argument('--win_size', default=7, type=int, help='window patch_size')
parser.add_argument('--embed_dim', default=96, type=int, help='embedding dimension')

# TRANSBTS
parser.add_argument('--patch_dim', default=[8, 8, 8], type=bool, help='window patch_size')

# nnFormer
parser.add_argument('--net_numpool', default=5, type=int)
parser.add_argument('--relative_mult', default=True, type=bool)
parser.add_argument("--depths", nargs="+", default=[2, 2, 2, 2])
parser.add_argument("--nn_former_num_heads", nargs="+", default=[3, 6, 12, 24])
parser.add_argument("--patch_size", nargs="+", default=[4, 4, 1])
parser.add_argument("--window_size", nargs="+", default=[[5, 5, 3], [5, 5, 3], [10, 10, 7], [5, 5, 3]])
parser.add_argument("--down_stride", nargs="+", default=[[4, 4, 1], [8, 8, 1], [16, 16, 2], [32, 32, 4]])
parser.add_argument("--ds_retained_len", default=5, help="number of scales used in ds (the value 1,1,1 counts"
                                                         " automatically as one)")
parser.add_argument("--deep_supervision_scales", nargs="+", default=None)

# nnUNet
parser.add_argument('--num_samples', default=None, type=int,
                    help='number of samples per epochs (250 for default nnUnet)')

# Early stopping
parser.add_argument('--use_eval_criterion', default=True,
                    help='Do not only focus on avg dice for determining the best model')
parser.add_argument('--eval_criterion_factor', default=0.9, help='Weight attributed to the past predictions')
parser.add_argument('--early_stopping', default=True, help='use early stopping')
parser.add_argument('--train_criterion_factor', default=0.9, help='Weight attributed to the past loss values')
parser.add_argument('--train_criterion_eps', default=5e-4, help='tolerance for train criterion comparison')
parser.add_argument('--lr_threshold', default=1e-6, help='Weight attributed to the past predictions')

# Data augmentation
# Images cropping
parser.add_argument('--crop', default='randByClass', type=str, help='random or fixed crop')
parser.add_argument('--roi_x', default=128, type=int, help='roi patch_size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi patch_size in y direction')
parser.add_argument('--roi_z', default=64, type=int, help='roi patch_size in z direction')
parser.add_argument('--num_samples_per_image', default=1, type=int, help='Num of used samples for each image by batch')

# Image flipping
parser.add_argument('--flip', default=True, type=bool, help='apply random flip in data augmentation')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')

# Image rotation
parser.add_argument('--rotate', default=True, type=bool, help='apply random rotation in data augmentation')
parser.add_argument('--rotation_range_x', default=np.pi / 4, type=bool, help='rotation range')
parser.add_argument('--rotation_range_y', default=np.pi / 4, type=bool, help='rotation range')
parser.add_argument('--rotation_range_z', default=np.pi / 4, type=bool, help='rotation range')
parser.add_argument('--RandRotated_prob', default=0.1, type=float, help='RandRotated aug probability')

# Image scaling
parser.add_argument('--scale', default=True, type=bool, help='apply scaling in data augmentation')
parser.add_argument('--randZoomd_prob', default=0.1, type=float, help='RandZoomd aug probability')

# Intensity scaling
parser.add_argument('--scale_intensity', default=True, type=bool,
                    help='apply random intensity scaling in data augmentation')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--scale_intensity_value', default=0.1, type=float, help='scaling value')

# Intensity shifting
parser.add_argument('--shift_intensity', default=True, type=bool,
                    help='apply random intensity shifting in data augmentation')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')

# Gaussian noise generation
parser.add_argument('--gaussian_noise', default=True, type=bool,
                    help='apply random gaussian noise in data augmentation')
parser.add_argument('--RandGaussianNoised_prob', default=0.1, type=float, help='RandGaussianNoisedaug probability')

# Gaussian blur generation
parser.add_argument('--gaussian_blur', default=True, type=bool, help='apply random gaussian blur in data augmentation')
parser.add_argument('--RandGaussianSmoothd_prob', default=0.1, type=float, help='RandGaussianSmoothd aug probability')

# Low resolution simulation
parser.add_argument('--simulate_low_resolution', default=True, type=bool,
                    help='apply low resolution simulation in data augmentation')
parser.add_argument('--SimulateLowResolutionTransform_prob', default=0.1, type=float,
                    help='SimulateLowResolutionTransform aug probability')

# Contrast transform
parser.add_argument('--adjust_contrast', default=True, type=bool,
                    help='apply random contrast adjustment in data augmentation')
parser.add_argument('--RandAdjustContrastd_prob', default=0.1, type=float, help='RandAdjustContrastd aug probability')

# Gamma transform
parser.add_argument('--gamma_transform', default=True, type=bool,
                    help='apply random gamma transformation in data augmentation')
parser.add_argument('--GammaTransform_prob', default=0.1, type=float, help='GammaTransform aug probability')
parser.add_argument('--gamma_range', default=(0.7, 1.5), type=tuple, help='GammaTransform aug range')
parser.add_argument('--gamma_retain_stats', default=0.1, type=float,
                    help='conserve initial mean and std after GammaTransform')

# Post training inference
parser.add_argument('--get_tumor_burden', default=True, help='calculate tumor burden at the end of the training')
parser.add_argument('--save_val_pred', default=True, type=bool,
                    help='after training save predictions  as nifti on validation dataset')
parser.add_argument('--save_to_original_spacing', default=False, type=bool,
                    help='should the saved predictions be resampled with to the original spacing of the images')
# Post transforms
parser.add_argument('--do_sigmoid', default=False, type=bool)

# Sweep
parser.add_argument('--project', default="sweeps-patch_size", type=str, help='wandb project')
parser.add_argument('--root', default="./config/pre-processing_and_data_augmentation/", type=str,
                    help='root to sweep configuration')
parser.add_argument('--file', default="optimisation_config.yaml", type=str, help='sweep optimisation file')
parser.add_argument('--sweep_id', default=None, type=str, help='sweep id')
parser.add_argument('--count', default=None, type=int, help='number of successive model to train')
parser.add_argument('--use_local_controller', default=True, type=bool, help='use classic or local controller')
parser.add_argument('--use_customized_bayes', default=True, type=bool,
                    help='use our bayes search implementation or not')

# Data augmentation sweep
parser.add_argument('--scaling_and_resolution_aug', default=True, type=bool, help='apply spatial augments or not')
parser.add_argument('--image_fidelity_aug', default=True, type=bool, help='apply gaussian augments or not')
parser.add_argument('--luminance_and_contrat_aug', default=True, type=bool, help='apply intensity augments or not')

# Learning paradigms sweep
parser.add_argument('--optimizer_config', default=None, nargs="+",
                    help='which optimizer/lr/weight decay to use for sweep config')

# Resume a run
parser.add_argument('--resume', default=False, help='resume a run or no')
parser.add_argument('--id_run', default=None, type=str, help='resume a run or no')


# to resume a run: resume=True, id_run with the concerned id and add checkpoint in the code juste before the
# args.checkpoint load to resume an offline run: args.resume = false


def main():
    args = parser.parse_args()

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        args = initialize_rank(args)
        if args.rank == 0:
            print('Found total gpus', args.ngpus_per_node)
            print("world patch_size:" + str(args.size))
            print("idrtorch patch_size:" + str(args.size))
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=args.size,
                                rank=args.rank)
    else:
        args.rank = 0

    if __name__ == "__main__":
        if args.config_file is not None:
            args = get_config(args)

    id = args.model_name + '-' + args.id
    if args.rank == 0:
        my_run = wandb.init(project=args.project, name=id, resume=args.resume, id=args.id_run,
                            group="DDP" if args.distributed else None)
        print("RUN ID: " + wandb.run._run_id)
        sweep_args = my_run.config

        if not __name__ == "__main__" or args.resume:
            args.config_file = sweep_args["config_file"]
            args = get_config(args)

        for key, value in sweep_args.items():
            args.__setattr__(key, value)

        if 'roi' in sweep_args.keys():
            args.roi_x = sweep_args['roi'][0]
            args.roi_y = sweep_args['roi'][1]
            args.roi_z = sweep_args['roi'][2]

        if args.optimizer_config is not None:
            args.optimizer = args.optimizer_config[0]
            args.optim_lr = args.optimizer_config[1]
            args.reg_weight = float(args.optimizer_config[2])

        my_run.config.update(args)

        args.nb_epoch_for_patience = args.epochs * 0.05

    else:
        my_run = None
    if args.distributed:
        args = [args]
        dist.broadcast_object_list(args, src=0)
        args = args[0]
        args = initialize_rank(
            args)  # here we need to call this function again because the previous line re-write the rank of each gpu

    if args.personalized_dir:
        date = datetime.now().strftime("%d_%m_%Y_%H:%M")
        if args.rank == 0:
            print(f"THE DATE AT THE BEGINNING OF THE TRAINING IS {date}")
        if args.id_dir:
            args.logdir = os.path.join(args.data_dir, 'results', id, date, 'fold_' + str(args.fold_id))
            model_path = os.path.join(args.data_dir, 'results', id, date)
            model_name_dir = os.path.join(args.data_dir, 'results', id)
        else:
            args.logdir = os.path.join(args.data_dir, 'results', args.model_name, date, 'fold_' + str(args.fold_id))
            model_path = os.path.join(args.data_dir, 'results', args.model_name, date)
            model_name_dir = os.path.join(args.data_dir, 'results', args.model_name)
    else:
        args.logdir = os.path.join(args.data_dir, 'results', args.model_name, 'fold_' + str(args.fold_id))
        model_path = os.path.join(args.data_dir, 'results', args.model_name)

    if args.rank == 0:
        if args.personalized_dir:
            try:
                os.mkdir(model_name_dir)
            except FileExistsError:
                pass
        try:
            os.mkdir(model_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(args.logdir)
        except FileExistsError:
            pass

    # Seeding process
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda")
    else:
        device = torch.device(args.device)

    args.test_mode = False
    loss = get_loss(args)

    if args.do_ds:
        args.deep_supervision_scales = [[1, 1, 1]] + args.deep_supervision_scales

    start_epoch = 1
    model = get_model(args)

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.', '')] = v
        model.load_state_dict(new_state_dict, strict=False)
        saved_values = checkpoint['saved_values']
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'optimizer' in checkpoint:
            for k, v in checkpoint['optimizer'].items():
                new_state_dict[k.replace('backbone.', '')] = v
            optimizer.load_state_dict(new_state_dict)
            optimizer_to_device(optimizer, device)
            if args.poly_learning_rate == True:
                poly_learning_rate(optimizer, start_epoch, args.epochs, args.optim_lr, exponent=0.9)
        if 'scheduler' in checkpoint:
            for k, v in checkpoint['scheduler'].items():
                new_state_dict[k.replace('backbone.', '')] = v
            scheduler.load_state_dict(new_state_dict)
        if args.rank == 0:
            print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, start_epoch))
    else:
        saved_values = None
        if args.rank == 0:
            print("=> No loaded checkpoint '{}' (epoch {})".format(args.checkpoint, start_epoch))

    if args.distributed:
        model = model.to(device)
        model = DDP(model, device_ids=[args.local_rank],
                    find_unused_parameters=True if args.model_name == "nnunet" or "transbts" else False)
    else:
        model.to(device)

    val_metric = GeneralizedDiceScore(include_background=False,
                                      reduction=MetricReduction.MEAN_BATCH)

    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = Compose([
        Activations(sigmoid=args.do_sigmoid),
        AsDiscrete(argmax=True,
                   to_onehot=args.out_channels
                   )])

    model_inferer = partial(sliding_window_inference,
                            roi_size=[args.roi_x, args.roi_y, args.roi_z],
                            scales=torch.tensor(
                                args.deep_supervision_scales[:args.ds_retained_len]) if args.do_ds else None,
                            sw_batch_size=args.batch_size,
                            predictor=model,
                            overlap=args.infer_overlap,
                            get_latent_space=False)

    train_loader, val_loader = get_loader(args)

    if args.rank == 0:
        print(f"Train dataset number of batch: {len(train_loader)}")
        print(f"Val dataset number of batch: {len(val_loader)}")
        print("start training now!")

    best_criterion = run_training(model=model,
                                  train_loader=train_loader,
                                  val_loader=val_loader,
                                  optimizer=optimizer,
                                  loss_func=loss,
                                  val_metric=val_metric,
                                  args=args,
                                  model_inferer=model_inferer,
                                  scheduler=scheduler,
                                  post_label=post_label,
                                  post_pred=post_pred,
                                  start_epoch=start_epoch,
                                  saved_values=saved_values,
                                  device=device,
                                  my_run=my_run)

    print(f"Best validation criterion: {best_criterion}")


if __name__ == '__main__':
    main()
