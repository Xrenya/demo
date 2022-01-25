from copy import copy
from typing import List, NoReturn, Tuple

import numpy as np
import torch


class BBox:
    MODES = ["xyxy", "xywh", "ccwh"]

    def __init__(self, bbox,
                 mode: str = "xyxy",
                 copy: bool = False):
        """BBox configuration

        :param bbox:
        :param mode: ["xyxy", "xywh", "ccwh"]
        :param copy: return copy
        """
        if mode not in BBox.MODES:
            raise ValueError(
                f"Expected modes {BBox.MODES}, but got {mode}"
            )
        if bbox.shape[-1] != 4:
            raise ValueError(
                f"The last dimension of the array must be of size 4, "
                f"but the given array shape is {bbox.shape}"
            )
        if copy:
            bbox = bbox.copy()
        if mode == "xywh":
            bbox[..., 2:] -= bbox[..., :2]
        elif mode == "ccwh":
            bbox[..., :2] = (bbox[..., 2:] + bbox[..., :2]) / 2
            bbox[..., 2:] -= bbox[..., :2]
        self.bbox = bbox

    @property
    def shape(self) -> int:
        return self.bbox.shape[-1]

    @property
    def left(self) -> float:
        return self.bbox[..., 0]

    @property
    def top(self) -> float:
        return self.bbox[..., 1]

    @property
    def right(self) -> float:
        return self.bbox[..., 2]

    @property
    def bottom(self) -> float:
        return self.bbox[..., 3]

    @property
    def center(self) -> float:
        cxcy = self._center()
        return cxcy[..., 0], cxcy[..., 1]

    def _center(self) -> Tuple[float]:
        return (self.bbox[..., :2] + self.bbox[..., 2:]) / 2

    @property
    def size(self):
        size = self.bbox[..., 2:] - self.bbox[..., :2]
        return size[..., 0], size[..., 1]

    @property
    def area(self) -> float:
        width, height = self.size
        is_valid = (width > 0) and (height > 0)
        return width * height * is_valid

    def __repr__(self):
        return f"BBox({repr(self.bbox).replace('array', 'np.array')})"

    # def __str__(self) -> NoReturn:
    #     text = f"BBox(x0={self.left}, y0={self.top}, " \
    #            f"x1={self.right}, y1={self.bottom})"
    #     return text

    def __eq__(self, other) -> bool:
        return (self.bbox == other).all()

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item: int) -> float:
        return BBox(self.bbox[item])

    def __round__(self, decimals: int = 1):
        return BBox(self.bbox.round(decimals))

    def __and__(self, other):
        left_top = np.maximum(self.bbox[..., :2], other.bbox[..., :2])
        right_bottom = np.minimum(self.bbox[..., 2:], other.bbox[..., 2:])
        return BBox(np.concatenate((left_top, right_bottom)))

    def __iadd__(self, other):
        dx, dy = self.verify_value(other)
        self.bbox[..., 0::2] += dx
        self.bbox[..., 1::2] += dy
        return self

    def __isub__(self, other):
        dx, dy = self.verify_value(other)
        self.bbox[..., 0::2] -= dx
        self.bbox[..., 1::2] -= dy
        return self

    def __copy__(self):
        return BBox(self.bbox, copy=True)

    def __add__(self, other):
        copy_bbox = copy(self.bbox)
        copy_bbox += other
        return copy_bbox

    def __sub__(self, other):
        copy_bbox = copy(self.bbox)
        copy_bbox -= other
        return copy_bbox

    def _scale(self, scale) -> NoReturn:
        kx, ky = self.verify_value(scale)
        self.bbox[..., 0] = self.bbox[..., 0] * kx
        self.bbox[..., 1] = self.bbox[..., 1] * ky
        self.bbox[..., 2] = self.bbox[..., 2] * kx
        self.bbox[..., 3] = self.bbox[..., 3] * ky

    def __imul__(self, scale):
        self._scale(scale)
        return self

    def __mul__(self, scale):
        bbox = copy(self)
        bbox *= scale
        return bbox

    def __itruediv__(self, scale):
        scale = self.verify_value(scale)
        scale_inv = (1 / scale[0], 1 / scale[1])
        self *= scale_inv
        return self

    def __truediv__(self, scale):
        bbox = copy(self)
        bbox /= scale
        return bbox

    def is_bbox_inside(self, size):
        w, h = self.verify_value(size)
        left = self.left
        top = self.top
        bottom = self.bottom
        right = self.right
        return (left <= w - 1) & (top <= h - 1) & (right <= w - 1) & (bottom <= h - 1)

    @staticmethod
    def verify_value(other) -> Tuple[float]:
        try:
            if len(other) == 2:
                return tuple(other)
            else:
                raise ValueError(f"Object's length should be 2, "
                                 f"but got {len(other)}")
        except TypeError:
            return other, other

    def IoU(self, other) -> float:
        intersection_area = (self & other).area
        return intersection_area / (self.area + other.area - intersection_area)

    def to_list(self) -> List[float]:
        return list(self.bbox)

    def to_tensor(self) -> List[float]:
        return torch.tensor(self.bbox)

    def export(self) -> List[int]:
        bbox = list(map(int, self.bbox))
        return bbox
