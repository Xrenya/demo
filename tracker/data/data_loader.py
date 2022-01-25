from random import choices

from torch.utils.data import ConcatDataset


class TrackingDataset:
    def __init__(self, tracking_datasets,
                 target_preprocessing=None,
                 search_preprocessing=None,
                 train_transforms=None,
                 valid_transforms=None):
        self.videos = list(
            ConcatDataset([dataset for dataset in tracking_datasets])
        )

        self.frames = [frame for frame in self.videos]
        self.target_preprocessing = target_preprocessing
        self.search_preprocessing = search_preprocessing
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        frames = self.videos[item]

        target_frame, search_frame = choices(frames, k=2)

        if self.train_transforms is not None:
            transformed_target = self.train_transforms(
                image=target_frame.image,
                mask=target_frame.mask,
                bboxes=target_frame.bbox,
                bbox_classes=[0],
            )
            transformed_search = self.train_transforms(
                image=search_frame.image,
                mask=search_frame.mask,
                bboxes=search_frame.bbox,
                bbox_classes=[0],
            )
        else:
            pass

        top = int(transformed_target["bboxes"][0][0])
        left = int(transformed_target["bboxes"][0][1])
        bottom = int(transformed_target["bboxes"][0][2])
        right = int(transformed_target["bboxes"][0][3])

        target = transformed_target["image"][:, left:right, top:bottom]
        search = transformed_search["image"]
        mask = transformed_search["mask"]

        return {
            "search": search,
            "target": target,
            "bbox": transformed_search["bboxes"][0],
            "mask": mask
        }
