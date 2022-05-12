from os.path import dirname
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import os

from modules.augmentation.transformations import Augmentation, get_transform
from tqdm import tqdm


def read_filelist(filepath, common_classes, tgt_private_classes, aug_transform: Optional[Augmentation]):
    print(filepath)
    all_classes = {*common_classes, *tgt_private_classes}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.strip().split() for l in lines]

    def load_pil_image(samp_path):
        img = Image.open(os.path.join(dirname(filepath), samp_path))
        img.convert("RGB")
        return img

    transform_list = [transforms.Resize((256, 256))]
    if aug_transform is not None:
        transform_list.append(get_transform(aug_transform))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    transform = transforms.Compose(transform_list)

    lines = [(transform(load_pil_image(l[0])).numpy(), int(l[1])) for l in tqdm(lines) if int(l[1]) in all_classes]
    features, labels = np.array(list(map(lambda x: x[0], lines))), np.array(list(map(lambda x: x[1], lines)))
    return features, labels
