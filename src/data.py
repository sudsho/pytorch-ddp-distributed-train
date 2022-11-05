"""Imagenette data loading.

Imagenette is a 10-class subset of ImageNet maintained by fast.ai.
Smaller than full ImageNet so iteration is fast.
"""
import os
from torchvision import datasets, transforms


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"


def get_transforms(image_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])


def build_dataset(root, split="train", image_size=224):
    # expects imagenette2-320 unpacked under root with train/ and val/
    folder = os.path.join(root, "imagenette2-320", split)
    return datasets.ImageFolder(folder, transform=get_transforms(image_size, train=(split == "train")))
