import numpy as np


class BboxFrame:
    def __init__(self, image, bbox):
        self.image = image
        self.bbox = bbox

    @property
    def size(self):
        return self.image.shape[:2][::-1]

    @property
    def shape(self):
        return self.bbox.shape

    def __getitem__(self, item):
        image = self.image[item]
        bbox = self.bbox[item]
        return BboxFrame(image, bbox)

    def to(self, *args, **kwargs):
        image = self.image.to(*args, **kwargs)
        bbox = self.bbox.to(*args, **kwargs)
        return BboxFrame(image, bbox)

    @staticmethod
    def stack(frames):
        image = np.stack([frame.image for frame in frames])
        bbox = np.stack([frame.bbox for frame in frames])
        return BboxFrame(image, bbox)


class SegFrame:
    def __init__(self, image, mask, bbox=None):
        self.image = image
        self.mask = mask
        self.bbox = bbox

    @property
    def bbox(self):
        if self._bbox is not None:
            return self._bbox
        else:
            self._bbox = self.from_binary_mask(self.mask)
        return self._bbox

    @bbox.setter
    def bbox(self, bbox):
        self._bbox = bbox

    @staticmethod
    def from_binary_mask(masks):
        n = masks.shape[-1]

        bounding_boxes = np.zeros((n, 4))

        for index, mask in enumerate(masks.transpose(2, 0, 1)):
            y, x = np.where(mask != 0)

            bounding_boxes[index, 0] = np.min(x)
            bounding_boxes[index, 1] = np.min(y)
            bounding_boxes[index, 2] = np.max(x)
            bounding_boxes[index, 3] = np.max(y)
        return bounding_boxes

    def __getitem__(self, item):
        image = self.image[item]
        mask = self.mask[item]
        bbox = self.bbox[item]
        return SegFrame(image, mask, bbox)

    def to(self, *args, **kwargs):
        image = self.image.to(*args, **kwargs)
        mask = self.mask.to(*args, **kwargs)
        bbox = self.bbox.to(*args, **kwargs)
        return BboxFrame(image, mask, bbox)

    @staticmethod
    def stack(frames):
        image = np.stack([frame.image for frame in frames])
        mask = np.stack([frame.mask for frame in frames])
        bbox = np.stack([frame.bbox for frame in frames])
        return BboxFrame(image, mask, bbox)
