import PIL
import math
import torch
import random
import numpy as np
from functools import partial
from PIL import Image, ImageOps, ImageEnhance, ImageChops, ImageFilter

if hasattr(Image, "Resampling"):
    _RANDOM_INTERPOLATION = (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)
    _DEFAULT_INTERPOLATION = Image.Resampling.BICUBIC
else:
    _RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)
    _DEFAULT_INTERPOLATION = Image.BICUBIC

def _interpolation(kwargs):
    interpolation = kwargs.pop("resample", _DEFAULT_INTERPOLATION)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    return interpolation

def _check_args_tf(kwargs):
    if "fillcolor" in kwargs and Augmentations._PIL_VER < (5, 0):
        kwargs.pop("fillcolor")
    kwargs["resample"] = _interpolation(kwargs)

class Augments:

    @staticmethod
    def auto_contrast(img, **__):
        return ImageOps.autocontrast(img)

    @staticmethod
    def equalize(img, **__):
        return ImageOps.equalize(img)

    @staticmethod
    def invert(img, **__):
        return ImageOps.invert(img)

    @staticmethod
    def rotate(img, degrees, **kwargs):
        _check_args_tf(kwargs)
        if Augmentations._PIL_VER >= (5, 2):
            return img.rotate(degrees, **kwargs)
        if Augmentations._PIL_VER >= (5, 0):
            w, h = img.size
            post_trans = (0, 0)
            rotn_center = (w / 2.0, h / 2.0)
            angle = -math.radians(degrees)
            matrix = [
                round(math.cos(angle), 15),
                round(math.sin(angle), 15),
                0.0,
                round(-math.sin(angle), 15),
                round(math.cos(angle), 15),
                0.0,
            ]

            def transform(x, y, matrix):
                (a, b, c, d, e, f) = matrix
                return a * x + b * y + c, d * x + e * y + f

            matrix[2], matrix[5] = transform(
                -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
            )
            matrix[2] += rotn_center[0]
            matrix[5] += rotn_center[1]
            return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
        return img.rotate(degrees, resample=kwargs["resample"])

    @staticmethod
    def posterize(img, bits_to_keep, **__):
        if bits_to_keep >= 8:
            return img
        return ImageOps.posterize(img, bits_to_keep)

    @staticmethod
    def solarize(img, thresh, **__):
        return ImageOps.solarize(img, thresh)

    @staticmethod
    def solarize_add(img, add, thresh=128, **__):
        lut = []
        for i in range(256):
            if i < thresh:
                lut.append(min(255, i + add))
            else:
                lut.append(i)

        if img.mode in ("L", "RGB"):
            if img.mode == "RGB" and len(lut) == 256:
                lut = lut + lut + lut
            return img.point(lut)

        return img

    @staticmethod
    def contrast(img, factor, **__):
        return ImageEnhance.Contrast(img).enhance(factor)

    @staticmethod
    def color(img, factor, **__):
        return ImageEnhance.Color(img).enhance(factor)

    @staticmethod
    def brightness(img, factor, **__):
        return ImageEnhance.Brightness(img).enhance(factor)

    @staticmethod
    def sharpness(img, factor, **__):
        return ImageEnhance.Sharpness(img).enhance(factor)

    @staticmethod
    def gaussian_blur(img, factor, **__):
        img = img.filter(ImageFilter.GaussianBlur(radius=factor))
        return img

    @staticmethod
    def gaussian_blur_rand(img, factor, **__):
        radius_min = 0.1
        radius_max = 2.0
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(radius_min, radius_max * factor)))
        return img

    @staticmethod
    def desaturate(img, factor, **_):
        factor = min(1., max(0., 1. - factor))
        return ImageEnhance.Color(img).enhance(factor)

    @staticmethod
    def shear_x(img, factor, **kwargs):
        _check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)

    @staticmethod
    def shear_y(img, factor, **kwargs):
        _check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)

    @staticmethod
    def translate_x_rel(img, pct, **kwargs):
        pixels = pct * img.size[0]
        _check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)

    @staticmethod
    def translate_y_rel(img, pct, **kwargs):
        pixels = pct * img.size[1]
        _check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)

    @staticmethod
    def translate_x_abs(img, pixels, **kwargs):
        _check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)

    @staticmethod
    def translate_y_abs(img, pixels, **kwargs):
        _check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)

    @staticmethod
    def _randomly_negate(v):
        return -v if random.random() > 0.5 else v

    @staticmethod
    def _rotate_level_to_arg(level, _hparams):
        level = (level / Augmentations._LEVEL_DENOM) * 30.
        level = Augments._randomly_negate(level)
        return level,

    @staticmethod
    def _enhance_level_to_arg(level, _hparams):
        return (level / Augmentations._LEVEL_DENOM) * 1.8 + 0.1,

    @staticmethod
    def _enhance_increasing_level_to_arg(level, _hparams):
        level = (level / Augmentations._LEVEL_DENOM) * .9
        level = max(0.1, 1.0 + Augments._randomly_negate(level))
        return level,

    @staticmethod
    def _minmax_level_to_arg(level, _hparams, min_val=0., max_val=1.0, clamp=True):
        level = (level / Augmentations._LEVEL_DENOM)
        level = min_val + (max_val - min_val) * level
        if clamp:
            level = max(min_val, min(max_val, level))
        return level,

    @staticmethod
    def _shear_level_to_arg(level, _hparams):
        level = (level / Augmentations._LEVEL_DENOM) * 0.3
        level = Augments._randomly_negate(level)
        return level,

    @staticmethod
    def _translate_abs_level_to_arg(level, hparams):
        translate_const = hparams['translate_const']
        level = (level / Augmentations._LEVEL_DENOM) * float(translate_const)
        level = Augments._randomly_negate(level)
        return level,

    @staticmethod
    def _translate_rel_level_to_arg(level, hparams):
        translate_pct = hparams.get('translate_pct', 0.45)
        level = (level / Augmentations._LEVEL_DENOM) * translate_pct
        level = Augments._randomly_negate(level)
        return level,

    @staticmethod
    def _posterize_level_to_arg(level, _hparams):
        return int((level / Augmentations._LEVEL_DENOM) * 4),

    @staticmethod
    def _posterize_increasing_level_to_arg(level, hparams):
        return 4 - Augments._posterize_level_to_arg(level, hparams)[0],

    @staticmethod
    def _posterize_original_level_to_arg(level, _hparams):
        return int((level / Augmentations._LEVEL_DENOM) * 4) + 4,

    @staticmethod
    def _solarize_level_to_arg(level, _hparams):
        return min(256, int((level / Augmentations._LEVEL_DENOM) * 256)),

    @staticmethod
    def _solarize_increasing_level_to_arg(level, _hparams):
        return 256 - Augments._solarize_level_to_arg(level, _hparams)[0],

    @staticmethod
    def _solarize_add_level_to_arg(level, _hparams):
        return min(128, int((level / Augmentations._LEVEL_DENOM) * 110)),

    @staticmethod
    def rand_bbox(img_shape, lam, margin=0., count=None):
        ratio = np.sqrt(1 - lam)
        img_h, img_w = img_shape[-2:]
        cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
        margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
        cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
        cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
        yl = np.clip(cy - cut_h // 2, 0, img_h)
        yh = np.clip(cy + cut_h // 2, 0, img_h)
        xl = np.clip(cx - cut_w // 2, 0, img_w)
        xh = np.clip(cx + cut_w // 2, 0, img_w)
        return yl, yh, xl, xh

    @staticmethod
    def rand_bbox_minmax(img_shape, minmax, count=None):
        assert len(minmax) == 2
        img_h, img_w = img_shape[-2:]
        cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
        cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
        yl = np.random.randint(0, img_h - cut_h, size=count)
        xl = np.random.randint(0, img_w - cut_w, size=count)
        yu = yl + cut_h
        xu = xl + cut_w
        return yl, yu, xl, xu

    @staticmethod
    def one_hot(x, num_classes, on_value=1., off_value=0.):
        x = x.long().view(-1, 1)
        return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)
    

class AugmentOp:

    def __init__(
            self,
            name,
            prob=0.5,
            magnitude=10,
            parameters=None
    ):
        
        self.name = name
        self.aug_fn = Augmentations.name2Op(name)
        self.level_fn = Augmentations.level2Arg(name)
        self.prob = prob
        self.magnitude = magnitude
        self.parameters = parameters

        self.kwargs = dict(
            fillcolor=parameters["img_mean"] if "img_mean" in parameters else Augmentations._FILL,
            resample=parameters["interpolation"] if "interpolation" in parameters else _RANDOM_INTERPOLATION
        )

        self.magnitude_std = self.parameters.get("magnitude_std", 0)
        self.magnitude_max = self.parameters.get("magnitude_max", None)

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        
        magnitude = self.magnitude
        if self.magnitude_std > 0:
            if self.magnitude_std == float("inf"):
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        
        upper_bound = self.magnitude_max or Augmentations._LEVEL_DENOM
        magnitude = max(0., min(magnitude, upper_bound))
        level_args = self.level_fn(magnitude, self.parameters) if self.level_fn is not None else tuple()

        return self.aug_fn(img, *level_args, **self.kwargs)

    def __repr__(self):
        fs = self.__class__.__name__ + f"(name={self.name}, p={self.prob}"
        fs += f", m={self.magnitude}, mstd={self.magnitude_std}"
        if self.magnitude_max is not None:
            fs += f", mmax={self.magnitude_max}"
        fs += ')'
        return fs


class Augmentations:

    _PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])
    _FILL = (128, 128, 128)
    _LEVEL_DENOM = 10.
    _RAND_INCREASING_TRANSFORMS = [
        "AutoContrast",
        "Equalize",
        "Invert",
        "Rotate",
        "PosterizeIncreasing",
        "SolarizeIncreasing",
        "SolarizeAdd",
        "ColorIncreasing",
        "ContrastIncreasing",
        "BrightnessIncreasing",
        "SharpnessIncreasing",
        "ShearX",
        "ShearY",
        "TranslateXRel",
        "TranslateYRel",
    ]

    _RAND_TRANSFORMS = [
        "AutoContrast",
        "Equalize",
        "Invert",
        "Rotate",
        "Posterize",
        "Solarize",
        "SolarizeAdd",
        "Color",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateXRel",
        "TranslateYRel",
    ]

    _NAME_TO_OP = {
        "AutoContrast": Augments.auto_contrast,
        "Equalize": Augments.equalize,
        "Invert": Augments.invert,
        "Rotate": Augments.rotate,
        "Posterize": Augments.posterize,
        "PosterizeIncreasing": Augments.posterize,
        "PosterizeOriginal": Augments.posterize,
        "Solarize": Augments.solarize,
        "SolarizeIncreasing": Augments.solarize,
        "SolarizeAdd": Augments.solarize_add,
        "Color": Augments.color,
        "ColorIncreasing": Augments.color,
        "Contrast": Augments.contrast,
        "ContrastIncreasing": Augments.contrast,
        "Brightness": Augments.brightness,
        "BrightnessIncreasing": Augments.brightness,
        "Sharpness": Augments.sharpness,
        "SharpnessIncreasing": Augments.sharpness,
        "ShearX": Augments.shear_x,
        "ShearY": Augments.shear_y,
        "TranslateX": Augments.translate_x_abs,
        "TranslateY": Augments.translate_y_abs,
        "TranslateXRel": Augments.translate_x_rel,
        "TranslateYRel": Augments.translate_y_rel,
        "Desaturate": Augments.desaturate,
        "GaussianBlur": Augments.gaussian_blur,
        "GaussianBlurRand": Augments.gaussian_blur_rand
    }

    _LEVEL_TO_ARG = {
        "AutoContrast": None,
        "Equalize": None,
        "Invert": None,
        "Rotate": Augments._rotate_level_to_arg,
        "Posterize": Augments._posterize_level_to_arg,
        "PosterizeIncreasing": Augments._posterize_increasing_level_to_arg,
        "PosterizeOriginal": Augments._posterize_original_level_to_arg,
        "Solarize": Augments._solarize_level_to_arg,
        "SolarizeIncreasing": Augments._solarize_increasing_level_to_arg,
        "SolarizeAdd": Augments._solarize_add_level_to_arg,
        "Color": Augments._enhance_level_to_arg,
        "ColorIncreasing": Augments._enhance_increasing_level_to_arg,
        "Contrast": Augments._enhance_level_to_arg,
        "ContrastIncreasing": Augments._enhance_increasing_level_to_arg,
        "Brightness": Augments._enhance_level_to_arg,
        "BrightnessIncreasing": Augments._enhance_increasing_level_to_arg,
        "Sharpness": Augments._enhance_level_to_arg,
        "SharpnessIncreasing": Augments._enhance_increasing_level_to_arg,
        "ShearX": Augments._shear_level_to_arg,
        "ShearY": Augments._shear_level_to_arg,
        "TranslateX": Augments._translate_abs_level_to_arg,
        "TranslateY": Augments._translate_abs_level_to_arg,
        "TranslateXRel": Augments._translate_rel_level_to_arg,
        "TranslateYRel": Augments._translate_rel_level_to_arg,
        "Desaturate": partial(Augments._minmax_level_to_arg, min_val=0.5, max_val=1.0),
        "GaussianBlur": partial(Augments._minmax_level_to_arg, min_val=0.1, max_val=2.0),
        "GaussianBlurRand": Augments._minmax_level_to_arg,
    }

    @classmethod
    def getTransforms(cls, increasing):
        if increasing:
            return cls._RAND_INCREASING_TRANSFORMS
        else:
            return cls._RAND_TRANSFORMS

    @staticmethod
    def getRandomAugmentOps(magnitude, prob, transforms, parameters):
        return [
            AugmentOp(
                name, prob=prob, magnitude=magnitude, parameters=parameters
            ) for name in transforms
        ]

    @classmethod
    def name2Op(cls, name):
        return cls._NAME_TO_OP[name]
    
    @classmethod
    def level2Arg(cls, name):
        return cls._LEVEL_TO_ARG[name]

    @staticmethod
    def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
        if ratio_minmax is not None:
            yl, yu, xl, xu = Augments.rand_bbox_minmax(img_shape, ratio_minmax, count=count)
        else:
            yl, yu, xl, xu = Augments.rand_bbox(img_shape, lam, count=count)
        if correct_lam or ratio_minmax is not None:
            bbox_area = (yu - yl) * (xu - xl)
            lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
        return (yl, yu, xl, xu), lam

    @staticmethod
    def mixup_target(target, num_classes, lam=1., smoothing=0.0, one_hot=False):
        off_value = smoothing / num_classes
        on_value = 1. - smoothing + off_value

        if not one_hot: return (target, target.flip(0), lam)

        y1 = Augments.one_hot(target, num_classes, on_value=on_value, off_value=off_value)
        y2 = Augments.one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value)

        return y1 * lam + y2 * (1. - lam)

    @staticmethod
    def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device="cuda"):
        if per_pixel:
            return torch.empty(patch_size, dtype=dtype, device=device).normal_()
        elif rand_color:
            return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
        else:
            return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)
