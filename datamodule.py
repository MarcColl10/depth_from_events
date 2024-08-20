from dataclasses import dataclass
from functools import partial
from pathlib import Path

from dotmap import DotMap
import h5py
import hdf5plugin
from lightning import LightningDataModule
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data_utils import batched, ConcatBatchSampler, time_first_collate


@dataclass
class FrameSequence:
    file: Path
    chunk_size: int = 100
    seq_len: int | None = None
    crop: tuple[int, ...] | None = None  # height, width or top, left, bottom, right
    augmentations: list[str] | None = None

    def __post_init__(self):
        # open large h5 files only once
        self.h5 = h5py.File(self.file, "r")
        self.sensor_size = (260, 346)  # TODO: add to h5 file attributes

        # get number of frames
        self.n_frames = len(self.h5["events/frames"])

        # slice dataset, pre-compute crop and augmentations
        self.reset()

        # mapping from chunks to individual frames
        # match seq_len if given
        self.chunk_size = self.seq_len if self.seq_len else self.chunk_size
        self.chunk_map = batched(range(len(self.slice)), self.chunk_size)

    def init_slice(self):
        if self.seq_len:  # randomly-sliced sequence of seq_len
            i_start = np.random.randint(0, self.n_frames - self.seq_len)
            self.slice = range(i_start, i_start + self.seq_len)
        else:  # full sequence
            self.slice = range(self.n_frames)

    def init_crop(self):
        if self.crop:
            if len(self.crop) == 2:  # height, width
                h, w = self.crop
                top = np.random.randint(self.sensor_size[0] - h + 1)  # +1 because exclusive
                left = np.random.randint(self.sensor_size[1] - w + 1)
                self.crop_ = (top, left, top + h, left + w)
            elif len(self.crop) == 4:  # top, left, bottom, right
                self.crop_ = self.crop
        else:
            self.crop_ = (0, 0, *self.sensor_size)

    def init_augmentation(self):
        self.augmentation = []
        if self.augmentations:
            for aug in self.augmentations:
                if np.random.rand() < 0.5:
                    self.augmentation.append(aug)

    def reset(self):
        self.init_slice()  # slice up dataset
        self.init_crop()  # pre-compute crop
        self.init_augmentation()  # pre-compute augmentations

    def __len__(self):
        return len(self.chunk_map)

    def __getitem__(self, idx):
        # get new random slice, crop, augmentations
        self.reset()

        # get frames
        # >6x faster than in a loop
        chunk = self.chunk_map[idx]
        start, stop = chunk[0], chunk[-1] + 1
        frames = torch.from_numpy(self.h5["events/frames"][start:stop].astype(np.float32))

        # crop
        top, left, bottom, right = self.crop_
        frames = frames[..., top:bottom, left:right]

        # apply augmentations
        if "flip_t" in self.augmentation:
            frames = frames.flip(0)
            frames[:, -1] = 1 - frames[:, -1]  # revert avg ts in [0, 1]
        if "flip_pol" in self.augmentation:
            frames[:, :2] = frames[:, :2].flip(1)  # only neg, pos
        if "flip_ud" in self.augmentation:
            frames = frames.flip(2)
        if "flip_lr" in self.augmentation:
            frames = frames.flip(3)

        # adapt camera matrices to crop and augmentations
        # TODO: store in h5 file or load from yaml
        K_rect = torch.tensor([[346, 0, 173, 0], [0, 346, 130, 0], [0, 0, 1, 0]], dtype=torch.float32)
        inv_K_rect = torch.linalg.pinv(K_rect)

        # return dotmap
        sample = DotMap()
        sample.frames = frames
        sample.recording = None
        sample.eof = [i == len(self.slice) - 1 for i in chunk]
        sample.K_rect = K_rect
        sample.inv_K_rect = inv_K_rect

        return sample


class DataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        train_seq_len,
        train_crop,
        augmentations,
        batch_size,
        shuffle,
        num_workers,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.train_seq_len = train_seq_len
        self.train_crop = train_crop
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def prepare_data(self):
        self.recordings = [
            "indoor_forward_3_davis_with_gt",
            "indoor_forward_5_davis_with_gt",
            "indoor_forward_6_davis_with_gt",
            "indoor_forward_7_davis_with_gt",
            "indoor_forward_8_davis",
            "indoor_forward_9_davis_with_gt",
            "indoor_forward_10_davis_with_gt",
            "indoor_forward_11_davis",
            "indoor_forward_12_davis",
        ]

    def setup(self, stage):
        sequence = partial(
            FrameSequence,
            seq_len=self.train_seq_len,
            crop=self.train_crop,
            augmentations=self.augmentations,
        )
        if stage == "fit":
            train_recordings = []
            for rec in self.recordings:
                name = ("_").join(rec.split("_")[:2])
                fname = self.root_dir / name / f"{rec}.h5"
                seq = sequence(file=fname)
                train_recordings.extend([fname] * int(seq.n_frames / seq.seq_len))
            self.train_dataset = ConcatDataset([sequence(file=fname) for fname in train_recordings])

    def train_dataloader(self):
        sampler = ConcatBatchSampler(self.train_dataset, self.batch_size, shuffle=self.shuffle)
        dataloader = DataLoader(
            self.train_dataset, batch_sampler=sampler, num_workers=self.num_workers, collate_fn=time_first_collate
        )
        return dataloader


if __name__ == "__main__":
    from rich.progress import track

    datamodule = DataModule(
        root_dir="data/uzh_fpv_10ms_0.25ts_rect",
        train_seq_len=100,
        train_crop=(128, 128),
        augmentations=["flip_t", "flip_pol", "flip_ud", "flip_lr"],
        batch_size=8,
        shuffle=True,
        num_workers=8,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()

    for _ in range(5):
        for batch in track(dataloader):
            pass
