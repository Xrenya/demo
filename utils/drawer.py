#!/usr/bin/env python3
from typing import Tuple

import cv2
import numpy as np


class Drawer:
    def __init__(self, image=None, color_base: Tuple[int] = (255, 255, 255),
                 colors_preds: Tuple[int] = (0, 0, 255),
                 mask_opacity: float = 0.5,
                 thickness: int = 1,
                 fontFace: int = cv2.FONT_HERSHEY_SIMPLEX,
                 fontScale: float = 0.5):
        self.image = image
        self.color_base = color_base
        self.colors_preds = colors_preds
        self.mask_opacity = mask_opacity
        self.thickness = thickness
        self.fontFace = fontFace
        self.fontScale = fontScale

    def draw_bbox(self, bbox_gt=None, bbox_pred=None,
                  name: str = "", score: float = None):
        if bbox_gt is not None:
            bbox_gt = bbox_gt.export()
            print(bbox_gt, bbox_gt[:2])
            cv2.rectangle(self.image, tuple(bbox_gt[:2]), tuple(bbox_gt[2:]),
                          self.color_base, self.thickness)
        if bbox_pred is not None:
            bbox = bbox_pred.export()
            cv2.rectangle(self.image, tuple(bbox[:2]), tuple(bbox[2:]),
                          self.colors_preds, self.thickness)
        if name:
            text = name
            if score is not None:
                text += f"{score:.2f}"
            if bbox_gt:
                cv2.putText(self.image, text, tuple(bbox_gt[:2]),
                            self.fontFace, self.fontScale,
                            self.color_base, self.thickness)
            else:
                cv2.putText(self.image, text, tuple(bbox[:2]),
                            self.fontFace, self.fontScale,
                            self.colors_preds, self.thickness)

    def draw_mask(self, mask=None):

        if mask is not None:
            red_mask = np.zeros(self.image.shape, dtype=self.image.dtype)
            red_mask[:, :, 0] = mask[:, :, 0]
            self.image = cv2.addWeighted(self.image, 1, red_mask,
                                         self.mask_opacity, 0)

    def write_frame_num(self, frame_num):
        frame_num_str = str(frame_num)
        (w, h), _ = cv2.getTextSize(frame_num_str, self.fontFace,
                                    self.fontScale, self.thickness)
        height, width = self.image.shape[:2]
        org = width - w - h, height - h
        cv2.putText(self.image, frame_num_str, org, self.fontFace,
                    self.fontScale, self.color_base, self.thickness)

