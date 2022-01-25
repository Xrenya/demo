from contextlib import contextmanager

import cv2
import numpy as np

from tracker.utils.functional import get_images, get_image_size, load_txt
from tracker.utils.frame import BboxFrame, SegFrame


class Video:
    def __init__(self, name: str, img_path: str,
                 bbox_path: str = None, mask_path: str = None):
        self.name = name
        self.img = get_images(img_path, ext="jpg")
        self.size = get_image_size(self.img[0])
        self.bbox_gt = None
        self.mask = None
        if bbox_path:
            self.bbox_gt = load_txt(bbox_path)
        if mask_path:
            self.mask = get_images(mask_path, ext="png")

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item: int):
        image = self.img[item]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.bbox_gt and self.mask is None:
            bbox_gt = self.bbox_gt[item]
            return BboxFrame(image, bbox_gt)
        else:
            if self.bbox_gt:
                bbox_gt = self.bbox_gt[item]
            else:
                bbox_gt = None
            mask_gt = self.mask[item]
            mask_gt = cv2.imread(mask_gt, cv2.IMREAD_GRAYSCALE)
            mask_gt = mask_gt[..., np.newaxis]
            return SegFrame(image, mask_gt, bbox_gt)

    @contextmanager
    def video_writer(self, filename: str, codec: str, fps: int):
        if filename is None:
            pass
        else:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(filename, fourcc, fps, self.size)
            try:
                yield video_writer
            finally:
                video_writer.release()

    def run(self, tracker=None, drawer=None, segmentation: bool = False,
            frame_num: bool = True,
            filename: str = None, codec: str = "mp4v", fps: int = 30,
            imshow: bool = False, delay: int = 20, tqdm = None):

        result_init = {}
        result_track = {}
        with self.video_writer(filename, codec, fps) as vw:
            for i, frame in enumerate(tqdm(self, desc=self.name)):
                if tracker is not None:
                    frame_ = frame.to(tracker.device, copy=True)
                    if i == 0:
                        _, duration = tracker.initialize_timed(frame_)
                        result_init["time"] = duration
                    else:
                        result, extra, duration = tracker.track_timed(frame_)
                        result["time"] = duration
                        for key, value in result.items():
                            if i == 1:
                                result_track[key] = [value]
                            else:
                                result_track[key].append(value)

                if drawer is not None:
                    frame_img = frame.image
                    frame_bbox = frame.bbox
                    drawer.image = frame_img
                    drawer.draw_bbox(bbox_gt=frame_bbox, name="GT")
                    if segmentation:
                        frame_mask = frame.mask
                        drawer.draw_mask(mask=frame_mask)

                vw.write(drawer.image)

                if imshow:
                    cv2.imshow(self.name, drawer.image)
                    cv2.waitKey(delay)

        cv2.destroyAllWindows()

        return {'init': result_init, 'track': result_track}


