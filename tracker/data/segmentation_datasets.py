from abc import ABC, abstractmethod
from json import load
from pathlib import Path
from typing import NoReturn

from tracker.utils.functional import get_images
from tracker.utils.video import Video


class SegmentationDataset(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__

    def __len__(self):
        return len(self.video_kwargs)

    def __getitem__(self, idx):
        return Video(**self.video_kwargs[idx])


class DAVIS(SegmentationDataset):
    def __init__(self,
                 split: str = "train",
                 root: str = "datasets/DAVIS") -> NoReturn:
        '''DAVIS Dataset: 2017

        :param split: dataset split on 'train' or 'val'
        :param root: dataset path
        '''
        super().__init__()
        if split not in ["train", "val"]:
            raise ValueError(
                f"Available splits: 'train' and 'val', but got {split}"
            )
        video_kwargs = []
        root = Path(root)
        img_root = root / "JPEGImages" / "1080p"
        mask_root = root / "Annotations" / "1080p"
        for video_root in img_root.iterdir():
            if video_root.is_dir():
                name = video_root.name
                video_name = img_root / name
                video_mask_name = mask_root / name
                video_kwargs.append(
                    {
                        "name": name,
                        "img_path": video_name,
                        "mask_path": video_mask_name
                     }
                )
        self.video_kwargs = video_kwargs


class YouTubeVOS(SegmentationDataset):
    def __init__(self, split, root='datasets/YouTube-VOS'):
        '''split = 'train' | 'valid' | 'test' '''
        super().__init__()
        root = Path(root)
        im_root = root / split / 'JPEGImages'
        mask_root = root / split / 'Annotations'
        meta_json = root / split / 'meta.json'
        obj_id_table = [138, 169, 203, 179, 154, 143]
        args = []
        with meta_json.open() as f:
            json_dict = load(f)
            for name, data in json_dict['videos'].items():
                im_dir = im_root / name
                mask_dir = mask_root / name
                obj_data = data['objects']
                frame_nums = [
                    p.stem for p in get_images(mask_dir, im_ext='.png')
                ]
                obj_frame_nums = {
                    int(k) - 1: [
                        frame_nums.index(fn)for fn in v['frames']
                    ]
                    for k, v in obj_data.items()
                }
                obj_dict = {
                    obj_id_table[k]: v for k, v in obj_frame_nums.items()
                }
                args.append((name, im_dir, mask_dir, obj_dict))
        self.args = args
