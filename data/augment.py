# List of augmentations based on randaugment
import random
import kornia
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from torchvision.transforms.transforms import Compose
import warnings
from typing import cast, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.functional import pad

from kornia.augmentation.base import GeometricAugmentationBase2D, IntensityAugmentationBase2D, TensorWithTransformMat
from kornia.color import rgb_to_grayscale
from kornia.constants import BorderType, pi, Resample, SamplePadding
from kornia.enhance import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    equalize,
    invert,
    posterize,
    sharpness,
    solarize,
)
from kornia.enhance.normalize import denormalize, normalize
from kornia.filters import box_blur, gaussian_blur2d, motion_blur
from kornia.geometry import (
    affine,
    crop_by_transform_mat,
    deg2rad,
    elastic_transform2d,
    get_affine_matrix2d,
    get_perspective_transform,
    get_tps_transform,
    hflip,
    remap,
    resize,
    vflip,
    warp_affine,
    warp_image_tps,
    warp_perspective,
)
from kornia.geometry.bbox import bbox_generator, bbox_to_mask
from kornia.geometry.transform.affwarp import _compute_rotation_matrix, _compute_tensor_center
from kornia.utils import _extract_device_dtype, create_meshgrid

from . import random_generator as rg
from .utils import _range_bound, _transform_input

random_mirror = True

def ShearX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def Identity(img, v):
    return img

def TranslateX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def TranslateXAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateYAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def Rotate(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Invert(img, _):
    return PIL.ImageOps.invert(img)

def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def Solarize(img, v):
    return PIL.ImageOps.solarize(img, v)

def Posterize(img, v):
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Color(img, v):
    return PIL.ImageEnhance.Color(img).enhance(v)

def Brightness(img, v):
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Sharpness(img, v):
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def augment_list():
    l = [
        (Identity, 0, 1),  
        (AutoContrast, 0, 1),
        (Equalize, 0, 1), 
        (Rotate, -30, 30),
        (Solarize, 0, 256),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Brightness, 0.05, 0.95),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.1, 0.1),
        (TranslateX, -0.1, 0.1),
        (TranslateY, -0.1, 0.1),
        (Posterize, 4, 8),
        (ShearY, -0.1, 0.1),
    ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

class Augment:
    def __init__(self, n):
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (random.random()) * float(maxval - minval) + minval
            img = op(img, val)

        return img

def get_augment(name):
    return augment_dict[name]

def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)

class Cutout(object):
    def __init__(self, n_holes, length, random=False):
        self.n_holes = n_holes
        self.length = length
        self.random = random

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        length = random.randint(1, self.length)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class RandomPerspective(GeometricAugmentationBase2D):
    r"""Applies a random perspective transformation to an image tensor with a given probability.

    .. image:: _static/img/RandomPerspective.png

    Args:
        p: probability of the image being perspectively transformed..
        distortion_scale: it controls the degree of distortion and ranges from 0 to 1.
        resample: the interpolation method to use.
        return_transform: if ``True`` return the matrix describing the transformation
                          applied to each.
        same_on_batch: apply the same transformation across the batch. Default: False.
        align_corners: interpolation flag.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.warp_pespective`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs= torch.tensor([[[[1., 0., 0.],
        ...                         [0., 1., 0.],
        ...                         [0., 0., 1.]]]])
        >>> aug = RandomPerspective(0.5, p=0.5)
        >>> out = aug(inputs)
        >>> out
        tensor([[[[0.0000, 0.2289, 0.0000],
                  [0.0000, 0.4800, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])
        >>> aug.inverse(out)
        tensor([[[[0.0500, 0.0961, 0.0000],
                  [0.2011, 0.3144, 0.0000],
                  [0.0031, 0.0130, 0.0053]]]])
    """

    def __init__(
        self,
        distortion_scale: Union[torch.Tensor, float] = 0.5,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False,
        same_on_batch: bool = False,
        align_corners: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super(RandomPerspective, self).__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, keepdim=keepdim
        )
        self._device, self._dtype = _extract_device_dtype([distortion_scale])
        self.distortion_scale = distortion_scale
        self.resample: Resample = Resample.get(resample)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            interpolation=torch.tensor(self.resample.value), align_corners=torch.tensor(align_corners)
        )

    def __repr__(self) -> str:
        repr = (
            f"distortion_scale={self.distortion_scale}, interpolation={self.resample.name}, "
            f"align_corners={self.align_corners}"
        )
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        distortion_scale = torch.as_tensor(self.distortion_scale, device=self._device, dtype=self._dtype)
        return rg.random_perspective_generator(
            batch_shape[0],
            batch_shape[-2],
            batch_shape[-1],
            distortion_scale,
            self.same_on_batch,
            self.device,
            self.dtype,
        )

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return get_perspective_transform(params['start_points'].to(input), params['end_points'].to(input))

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, _, height, width = input.shape
        transform = cast(torch.Tensor, transform)
        return warp_perspective(
            input, transform, (height, width), mode=self.resample.name.lower(), align_corners=self.align_corners
        )