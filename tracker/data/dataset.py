from random import choice, choices
from torch.utils.data import Dataset
from tracker.utils.frame import SegFrame, BboxFrame

class FramePairDataset(Dataset):
    def __init__(self, videos, target, search):
        super().__init__()
        self.videos = videos
        self.target = target
        self.search = search

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]

        z_frame, = choices(video, weights, k=1)


        with no_grad():
            _, z_frame = self.prep_z(z_frame)
            _, x_frame = self.prep_x(x_frame)
        return z_frame, x_frame

class SingleFrameDataset(Dataset):
    def __init__(self, videos, prep):
        super().__init__()
        self.videos = videos
        self.prep = prep

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        weights = video.sane.tolist()
        frame, = choices(video, weights, k=1)
        _, frames = self.prep(frame)
        return frames

class SingleFrameDatasetFromStill(SingleFrameDataset):
    def __getitem__(self, idx):
        im_path, bblist = self.videos[idx]
        image = read_image(str(im_path), mode=ImageReadMode.RGB)
        bbox = choice(bblist)
        frame = Frame(image, bbox)
        with no_grad():
            _, frames = self.prep(frame)
        return frames
