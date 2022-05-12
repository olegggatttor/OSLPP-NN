from enum import Enum
import torchvision.transforms as T


class Augmentation(Enum):
    TRANSLATE = 1
    ROTATE = 2
    FLIP = 3
    SCALE = 4
    AFFINE_DEGREES = 5
    BRIGHTNESS = 6
    CONTRAST = 7
    BLUR = 8


def get_transform(transform: Augmentation):
    if transform == Augmentation.TRANSLATE:
        return T.RandomAffine(degrees=0, translate=(0.3, 0.8))
    if transform == Augmentation.ROTATE:
        return T.RandomRotation(degrees=(0, 360))
    if transform == Augmentation.FLIP:
        return T.RandomApply(transforms=[T.RandomHorizontalFlip(), T.RandomVerticalFlip()], p=1)
    if transform == Augmentation.SCALE:
        return T.RandomAffine(degrees=0, scale=(0.6, 1.2))
    if transform == Augmentation.AFFINE_DEGREES:
        return T.RandomAffine(degrees=(0, 360), translate=(0.3, 0.8), scale=(0.6, 1.2))
    if transform == Augmentation.BRIGHTNESS:
        return T.ColorJitter(brightness=0.8)
    if transform == Augmentation.CONTRAST:
        return T.ColorJitter(contrast=0.8)
    if transform == Augmentation.BLUR:
        return T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    raise Exception("Unknown transformation.")
