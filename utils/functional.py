import logging
import os
import pathlib
import random
import re
from typing import List, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_images(path: str, ext: str = "jpg") -> List[str]:
    """Return all images with corresponding extension

    Args:
        path: image folder's path
        ext: image file format extension

    Returns:
        An array of images' paths
    """
    images = pathlib.Path(path).glob("*." + ext)
    images = sorted(images)
    images = list(map(str, images))
    return images


def get_image_size(img_path: str) -> Tuple[int, int]:
    """Return image height and width

    Args:
        img_path: image path

    Returns:
        Tuple of width and height of the image (w, h)
    """
    img = cv2.imread(img_path)
    h, w, c = img.shape
    return w, h


def load_txt(fname, dtype: str = "float32", delimiter: str = None,
             usecols: int = None, ndmin: int = 2, start: int = None,
             stop: int = None, step: int = None):
    if delimiter:
        bbox = np.loadtxt(fname, dtype, delimiter=delimiter,
                          usecols=usecols, ndmin=ndmin)
    else:
        with open(fname) as f:
            array = [re.sub(',|\t', ' ', line) for line in f]
        bbox = np.loadtxt(array, dtype, usecols=usecols, ndmin=ndmin)
    bbox = bbox[start:stop:step]
    return bbox


def random_seed(seed: int = 3407):
    """Fixing random seed

    Args:
        seed: random seed number

    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if not (min_seed_value <= seed <= max_seed_value):
        logger.warning(
            f"{seed} is not in bounds, "
            f"numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = random.randint(min_seed_value, max_seed_value)

    logger.info(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
