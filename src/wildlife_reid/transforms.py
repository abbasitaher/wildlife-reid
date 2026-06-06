from __future__ import annotations

import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    if height > width:
        pad_width = (height - width) // 2
        padding = (pad_width, 0, pad_width, 0)
    else:
        pad_height = (width - height) // 2
        padding = (0, pad_height, 0, pad_height)
    return T.functional.pad(image, padding, fill=0, padding_mode="reflect")


def build_eval_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(pad_to_square),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def apply_preprocess(image: Image.Image, image_size: int):
    return build_eval_transform(image_size)(image)


def apply_augmentation(image: Image.Image) -> Image.Image:
    augment = T.Compose(
        [
            T.RandomApply([T.ColorJitter(contrast=0.4)], p=0.5),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20, fill=(0,)),
            T.RandomApply([T.ColorJitter(brightness=0.2)], p=0.5),
        ]
    )
    return augment(image)
