from typing import Callable, Tuple, Union, Sequence

# from batchgenerators.transforms import AbstractTransform
import numpy as np
from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_gamma
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_blur, augment_gaussian_noise
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy
from batchgenerators.augmentations.spatial_transformations import augment_spatial, augment_mirroring
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from monai.utils import GridSampleMode, GridSamplePadMode
from monai.utils.enums import PostFix

DEFAULT_POST_FIX = PostFix.meta()
GridSampleModeSequence = Union[Sequence[Union[GridSampleMode, str]], GridSampleMode, str]
GridSamplePadModeSequence = Union[Sequence[Union[GridSamplePadMode, str]], GridSamplePadMode, str]


class DownsampleSegForDSTransform(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''

    def __init__(self, ds_scales=(1, 0.5, 0.25), order=0, cval=0, input_key="seg", output_key="seg", axes=None):
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.cval = cval
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, data_dict):
        data_dict[self.output_key] = downsample_seg_for_ds_transform(data_dict[self.input_key], self.ds_scales,
                                                                     self.order, self.axes)
        return data_dict


def downsample_seg_for_ds_transform(seg, ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), order=0,
                                    axes=None):
    if axes is None:
        axes = list(range(1, len(seg.shape)))
    output = []
    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(seg)
        else:
            new_shape = np.array(seg.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            new_shape = np.round(new_shape).astype(int)
            out_seg = np.zeros(new_shape, dtype=seg.dtype)
            for b in range(seg.shape[0]):
                out_seg[b] = resize_segmentation(seg[b], new_shape[1:], order)
            # out_seg = resize_segmentation(seg, new_shape[1:], order, cval)
            output.append(out_seg)
    return output


class Swap(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, data_dict):
        data_dict['image'] = np.swapaxes(data_dict['image'], 1, 3)
        data_dict['label'] = np.swapaxes(data_dict['label'], 1, 3)
        # for b in range(len(data_dict[self.data_key])):
        return data_dict


class GammaTransform(AbstractTransform):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="image",
                 retain_stats: Union[bool, Callable[[], bool]] = False, p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation. retain_stats
        can also be callable (signature retain_stats() -> bool)
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, data_dict):
        # for b in range(len(data_dict[self.data_key])):
        if np.random.uniform() < self.p_per_sample:
            data_dict[self.data_key] = augment_gamma(data_dict[self.data_key], self.gamma_range,
                                                     self.invert_image,
                                                     per_channel=self.per_channel,
                                                     retain_stats=self.retain_stats)
        return data_dict


class NumpyToTensor(AbstractTransform):
    def __init__(self, keys=None, cast_to=None):
        """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
        :param keys: specify keys to be converted to tensors. If None then all keys will be converted
        (if value id np.ndarray). Can be a key (typically string) or a list/tuple of keys
        :param cast_to: if not None then the values will be cast to what is specified here. Currently only half, float
        and long supported (use string)
        """
        if keys is not None and not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.cast_to = cast_to

    def cast(self, tensor):
        if self.cast_to is not None:
            if self.cast_to == 'half':
                tensor = tensor.half()
            elif self.cast_to == 'float':
                tensor = tensor.float()
            elif self.cast_to == 'long':
                tensor = tensor.long()
            else:
                raise ValueError('Unknown value for cast_to: %s' % self.cast_to)
        return tensor

    def __call__(self, data_dict):
        import torch

        if self.keys is None:
            for key, val in data_dict.items():
                if isinstance(val, np.ndarray):
                    data_dict[key] = self.cast(torch.from_numpy(val)).contiguous()
                elif isinstance(val, (list, tuple)) and all([isinstance(i, np.ndarray) for i in val]):
                    data_dict[key] = [self.cast(torch.from_numpy(i)).contiguous() for i in val]
        else:
            for key in self.keys:
                if isinstance(data_dict[key], np.ndarray):
                    data_dict[key] = self.cast(torch.from_numpy(data_dict[key])).contiguous()
                elif isinstance(data_dict[key], (list, tuple)) and all(
                        [isinstance(i, np.ndarray) for i in data_dict[key]]):
                    data_dict[key] = [self.cast(torch.from_numpy(i)).contiguous() for i in data_dict[key]]

        return data_dict


class SimulateLowResolutionTransform(AbstractTransform):
    """Downsamples each sample (linearly) by a random factor and upsamples to original resolution again
    (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel:

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:

        order_upsample:
    """

    def __init__(self, zoom_range=(0.5, 1), per_channel=False, p_per_channel=1,
                 channels=None, order_downsample=1, order_upsample=0, data_key="image", p_per_sample=1,
                 ignore_axes=None):
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.zoom_range = zoom_range
        self.ignore_axes = ignore_axes

    def __call__(self, data_dict):
        # for b in range(len(data_dict[self.data_key])):
        if np.random.uniform() < self.p_per_sample:
            data_dict[self.data_key] = augment_linear_downsampling_scipy(data_dict[self.data_key],
                                                                         zoom_range=self.zoom_range,
                                                                         per_channel=self.per_channel,
                                                                         p_per_channel=self.p_per_channel,
                                                                         channels=self.channels,
                                                                         order_downsample=self.order_downsample,
                                                                         order_upsample=self.order_upsample,
                                                                         ignore_axes=self.ignore_axes)
        return data_dict


class ContrastAugmentationTransform(AbstractTransform):
    def __init__(self,
                 contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
                 preserve_range: bool = True,
                 per_channel: bool = True,
                 data_key: str = "image",
                 p_per_sample: float = 1,
                 p_per_channel: float = 1):
        """
        Augments the contrast of data
        :param contrast_range:
            (float, float): range from which to sample a random contrast that is applied to the data. If
                            one value is smaller and one is larger than 1, half of the contrast modifiers will be >1
                            and the other half <1 (in the inverval that was specified)
            callable      : must be contrast_range() -> float
        :param preserve_range: if True then the intensity values after contrast augmentation will be cropped to min and
        max values of the data before augmentation.
        :param per_channel: whether to use the same contrast modifier for all color channels or a separate one for each
        channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, data_dict):
        if np.random.uniform() < self.p_per_sample:
            data_dict[self.data_key] = augment_contrast(data_dict[self.data_key],
                                                        contrast_range=self.contrast_range,
                                                        preserve_range=self.preserve_range,
                                                        per_channel=self.per_channel,
                                                        p_per_channel=self.p_per_channel)
        return data_dict


class SpatialTransform(AbstractTransform):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch patch_size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch patch_size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of patch_size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of patch_size patch_size

        independent_scale_for_each_axis: If True, a scale factor will be chosen independently for each axis.
    """

    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key="image",
                 label_key="label", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                 independent_scale_for_each_axis=False, p_rot_per_axis: float = 1,
                 p_independent_scale_per_axis: int = 1):
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.p_rot_per_axis = p_rot_per_axis
        self.p_independent_scale_per_axis = p_independent_scale_per_axis

    def __call__(self, data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial(data, seg, patch_size=patch_size,
                                  patch_center_dist_from_border=self.patch_center_dist_from_border,
                                  do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                  do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                  angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                  border_mode_data=self.border_mode_data,
                                  border_cval_data=self.border_cval_data, order_data=self.order_data,
                                  border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                  order_seg=self.order_seg, random_crop=self.random_crop,
                                  p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                  p_rot_per_sample=self.p_rot_per_sample,
                                  independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                  p_rot_per_axis=self.p_rot_per_axis,
                                  p_independent_scale_per_axis=self.p_independent_scale_per_axis)
        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]

        return data_dict


class GaussianNoiseTransform(AbstractTransform):
    def __init__(self, noise_variance=(0, 0.1), p_per_sample=1, p_per_channel: float = 1,
                 per_channel: bool = False, data_key="image"):
        """
        Adds additive Gaussian Noise

        :param noise_variance: variance is uniformly sampled from that range
        :param p_per_sample:
        :param p_per_channel:
        :param per_channel: if True, each channel will get its own variance sampled from noise_variance
        :param data_key:

        CAREFUL: This transform will modify the value range of your data!
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel

    def __call__(self, data_dict):
        if np.random.uniform() < self.p_per_sample:
            data_dict[self.data_key] = augment_gaussian_noise(data_dict[self.data_key], self.noise_variance,
                                                              self.p_per_channel, self.per_channel)
        return data_dict


class GaussianBlurTransform(AbstractTransform):
    def __init__(self, blur_sigma: Tuple[float, float] = (1, 5), different_sigma_per_channel: bool = True,
                 different_sigma_per_axis: bool = False, p_isotropic: float = 0, p_per_channel: float = 1,
                 p_per_sample: float = 1, data_key: str = "image"):
        """

        :param blur_sigma:
        :param data_key:
        :param different_sigma_per_axis: if True, anisotropic kernels are possible
        :param p_isotropic: only applies if different_sigma_per_axis=True, p_isotropic is the proportion of isotropic
        kernels, the rest gets random sigma per axis
        :param different_sigma_per_channel: whether to sample a sigma for each channel or all channels at once
        :param p_per_channel: probability of applying gaussian blur for each channel. Default = 1 (all channels are
        blurred with prob 1)
        """
        self.p_per_sample = p_per_sample
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.blur_sigma = blur_sigma
        self.different_sigma_per_axis = different_sigma_per_axis
        self.p_isotropic = p_isotropic

    def __call__(self, data_dict):
        if np.random.uniform() < self.p_per_sample:
            data_dict[self.data_key] = augment_gaussian_blur(data_dict[self.data_key], self.blur_sigma,
                                                             self.different_sigma_per_channel,
                                                             self.p_per_channel,
                                                             different_sigma_per_axis=self.different_sigma_per_axis,
                                                             p_isotropic=self.p_isotropic)
        return data_dict


class MirrorTransform(AbstractTransform):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes=(0, 1, 2), data_key="image", label_key="label", p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if np.random.uniform() < self.p_per_sample:
            sample_seg = None
            if seg is not None:
                sample_seg = seg
            ret_val = augment_mirroring(data, sample_seg, axes=self.axes)
            data = ret_val[0]
            if seg is not None:
                seg = ret_val[1]

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict
