import math
import os
from typing import Optional

import numpy as np
import torch.distributed as dist
from monai import transforms, data
from monai.data import load_decathlon_datalist
from sklearn.model_selection import KFold
from torch.utils.data import RandomSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

from loaders_and_transforms.custom_transform import (DownsampleSegForDSTransform, SimulateLowResolutionTransform,
                                                     GammaTransform, NumpyToTensor, ContrastAugmentationTransform)


class RandomDistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, num_samples=None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        self.num_samples = num_samples
        if self.num_samples is None:
            self.num_samples = len(self.dataset)
        indices = list(range(self.num_samples))
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.

            self.num_samples = math.ceil(
                (self.num_samples - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.num_samples / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        self.valid_length = len(indices[self.rank:self.total_size:self.num_replicas])


def get_loader(args):
    val_transform = val_transforms(args)

    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json,
                                             True,
                                             "test",
                                             base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_loader = data.DataLoader(test_ds,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=1,
                                      pin_memory=True,
                                      persistent_workers=True)
        loader = test_loader
    else:
        files = load_decathlon_datalist(datalist_json,
                                        True,
                                        "training",
                                        base_dir=data_dir)

        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        train_idx, val_idx = list(kf.split(files))[args.fold_id]
        train_files = [files[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]

        train_transform = train_transforms(args)

        train_transform.set_random_state(seed=args.seed)
        val_transform.set_random_state(seed=args.seed)

        if args.debug:
            val_ds = data.Dataset(data=val_files, transform=val_transform)
            train_ds = data.Dataset(data=train_files, transform=train_transform)
        else:
            val_ds = data.CacheDataset(
                data=val_files,
                transform=val_transform,
                cache_rate=1.0,
                cache_num=20,
                num_workers=5,
                #  copy_cache=True,
            )
            train_ds = data.CacheDataset(
                data=train_files,
                transform=train_transform,
                cache_rate=1.0,
                cache_num=20,
                num_workers=8,
                #   copy_cache=True,
            )

        sampler_val = None

        if args.distributed:
            sampler_val = RandomDistributedSampler(val_ds, num_replicas=args.size,
                                                   rank=args.rank,
                                                   shuffle=False)

        sampler = None
        if args.num_samples is not None:
            sampler = RandomSampler(train_ds,
                                    num_samples=args.num_samples
                                    )

        if args.distributed:
            sampler = RandomDistributedSampler(train_ds,
                                               num_replicas=args.size,
                                               num_samples=args.num_samples,
                                               rank=args.rank)
        if args.debug:
            val_loader = data.DataLoader(val_ds,
                                         batch_size=1,
                                         sampler=sampler_val,
                                         shuffle=False,
                                         num_workers=1,
                                         pin_memory=True,
                                         persistent_workers=True)
            train_loader = data.DataLoader(train_ds,
                                           shuffle=(sampler is None),
                                           batch_size=args.batch_size,
                                           sampler=sampler,
                                           num_workers=1,
                                           pin_memory=True,
                                           persistent_workers=True)
        else:
            val_loader = data.ThreadDataLoader(val_ds,
                                               batch_size=1,
                                               sampler=sampler_val,
                                               shuffle=False,
                                               num_workers=5,
                                               pin_memory=True,
                                               persistent_workers=False)

            train_loader = data.ThreadDataLoader(train_ds,
                                                 batch_size=args.batch_size,
                                                 num_workers=8,
                                                 buffer_size=1,
                                                 shuffle=(sampler is None),
                                                 sampler=sampler,
                                                 buffer_timeout=1.0,
                                                 persistent_workers=True,
                                                 pin_memory=True,
                                                 repeats=1)

        loader = [train_loader, val_loader]

    return loader


def train_transforms(args):
    all_keys = ["image", "label"]
    transforms_list = [transforms.LoadImaged(keys=all_keys), transforms.AddChanneld(keys=all_keys),
                       transforms.Orientationd(keys=all_keys, axcodes="RAS"), transforms.SpatialPadd(keys=all_keys,
                                                                                                     spatial_size=(
                                                                                                         args.roi_x,
                                                                                                         args.roi_y,
                                                                                                         args.roi_z))]

    if args.rotate:
        transforms_list.append(transforms.RandRotated(keys=all_keys,
                                                      mode=["bilinear", "nearest"],
                                                      range_x=args.rotation_range_x,
                                                      range_y=args.rotation_range_y, range_z=args.rotation_range_z,
                                                      prob=args.RandRotated_prob, keep_size=True,
                                                      align_corners=False, allow_missing_keys=False))

    if args.crop == "randByClass":
        transforms_list.append(transforms.RandCropByLabelClassesd(
            keys=all_keys,
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            ratios=[1, 1, 1] if args.out_channels == 3 else [1, 1],
            num_samples=args.num_samples_per_image,
            num_classes=args.out_channels,
            image_key="image"
        ))

    else:
        transforms_list.append(transforms.RandSpatialCropSamplesd(
            keys=all_keys,
            num_samples=args.num_samples_per_image,
            roi_size=(args.first_roi_x, args.first_roi_y, args.roi_z),
            random_size=False))

    if args.scale_intensity:
        transforms_list.append(transforms.RandScaleIntensityd(keys="image", factors=args.scale_intensity_value,
                                                              prob=args.RandScaleIntensityd_prob))

    if args.shift_intensity:
        transforms_list.append(transforms.RandShiftIntensityd(keys="image", offsets=0.1,
                                                              prob=args.RandShiftIntensityd_prob))
    if args.image_fidelity_aug:
        if args.gaussian_noise:
            transforms_list.append(
                transforms.RandGaussianNoised(keys="image", prob=args.RandGaussianNoised_prob, mean=0.0,
                                              std=0.1))

        if args.gaussian_blur:
            transforms_list.append(transforms.RandGaussianSmoothd(keys="image", sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0),
                                                                  sigma_z=(0.5, 1.0), approx='erf',
                                                                  prob=args.RandGaussianSmoothd_prob))
    if args.luminance_and_contrat_aug:
        if args.adjust_contrast:
            transforms_list.append(ContrastAugmentationTransform(p_per_sample=args.RandAdjustContrastd_prob))

        if args.gamma_transform:
            transforms_list.append(GammaTransform(args.gamma_range, True, True, retain_stats=True, p_per_sample=0.1))
            transforms_list.append(GammaTransform(args.gamma_range, False, True, retain_stats=True, p_per_sample=0.3))

    if args.scaling_and_resolution_aug:
        if args.simulate_low_resolution:
            transforms_list.append(
                SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                                               order_downsample=0, order_upsample=3,
                                               p_per_sample=args.SimulateLowResolutionTransform_prob,
                                               ignore_axes=(0,)))
        if args.scale:
            transforms_list.append(
                transforms.RandZoomd(keys=all_keys, prob=args.randZoomd_prob, min_zoom=0.7, max_zoom=1.4,
                                     mode=["trilinear", "nearest"],
                                     padding_mode='constant', align_corners=None, keep_size=True,
                                     allow_missing_keys=False))

    if args.flip:
        transforms_list.append(transforms.RandFlipd(keys=all_keys, prob=args.RandFlipd_prob, spatial_axis=0))
        transforms_list.append(transforms.RandFlipd(keys=all_keys, prob=args.RandFlipd_prob, spatial_axis=1))
        transforms_list.append(transforms.RandFlipd(keys=all_keys, prob=args.RandFlipd_prob, spatial_axis=2))

    if args.do_ds:
        deep_supervision_scales = list(list(i) for i in 1 / np.vstack(args.deep_supervision_scales))[:-1]

        transforms_list.append(
            DownsampleSegForDSTransform(deep_supervision_scales, 0, input_key='label', output_key='label'))

        transforms_list.append(NumpyToTensor(all_keys, 'float'))
    else:
        transforms_list.append(transforms.ToTensord(keys=all_keys))

    return transforms.Compose(transforms_list)


def val_transforms(args):
    all_keys = ["image", "label"]

    val_transforms_list = [transforms.LoadImaged(keys=all_keys), transforms.AddChanneld(keys=all_keys),
                           transforms.Orientationd(keys=all_keys,
                                                   axcodes="RAS"), transforms.SpatialPadd(keys=all_keys,
                                                                                          spatial_size=(
                                                                                              args.roi_x, args.roi_y,
                                                                                              args.roi_z))]

    if args.do_ds:
        deep_supervision_scales = list(list(i) for i in 1 / np.vstack(args.deep_supervision_scales))[:-1]
        val_transforms_list.append(
            DownsampleSegForDSTransform(deep_supervision_scales, 0, input_key='label', output_key='label'))
        val_transforms_list.append(NumpyToTensor(all_keys, 'float'))
    else:
        val_transforms_list.append(transforms.ToTensord(keys=all_keys))

    return transforms.Compose(val_transforms_list)
