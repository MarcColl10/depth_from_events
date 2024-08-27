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

from data_utils import batched, ConcatBatchSampler, only_add_batch_dim, time_first_collate


@dataclass
class FrameSequence:
    root_dir: Path
    recording: str
    chunk_size: int = 100
    seq_len: int | None = None
    crop: tuple[int, ...] | None = None  # height, width or top, left, bottom, right
    augmentations: list[str] | None = None

    def __post_init__(self):
        # open large h5 files only once
        self.h5 = h5py.File(self.root_dir / f"{self.recording}.h5", "r")
        self.sensor_size = self.h5.attrs["sensor_size"]
        self.K_rect = torch.from_numpy(self.h5.attrs["K_rect"].astype(np.float32))

        # get number of frames
        self.n_frames = len(self.h5["events/frames"])

        # slice dataset, pre-compute crop and augmentations
        self.reset()

        # set frame shape
        self.frame_shape = (
            self.crop_corners[2] - self.crop_corners[0],
            self.crop_corners[3] - self.crop_corners[1],
        )

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
                self.crop_corners = (top, left, top + h, left + w)
            elif len(self.crop) == 4:  # top, left, bottom, right
                self.crop_corners = self.crop
        else:
            self.crop_corners = (0, 0, *self.sensor_size)

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
        top, left, bottom, right = self.crop_corners
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
        K_rect = self.K_rect.clone()
        K_rect[0, 2] -= left
        K_rect[1, 2] -= top
        if "flip_ud" in self.augmentation:
            K_rect[1, 2] = (bottom - top - 1) - K_rect[1, 2]
        if "flip_lr" in self.augmentation:
            K_rect[0, 2] = (right - left - 1) - K_rect[0, 2]
        inv_K_rect = torch.linalg.pinv(K_rect)

        # return dotmap
        sample = DotMap()
        sample.frames = frames
        sample.recording = self.recording
        sample.eofs = [i == len(self.slice) - 1 for i in chunk]
        sample.K_rect = K_rect
        sample.inv_K_rect = inv_K_rect

        return sample


class DataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        train_seq_len,
        train_crop,
        val_crop,
        augmentations,
        batch_size,
        shuffle,
        num_workers,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.train_seq_len = train_seq_len
        self.train_crop = train_crop
        self.val_crop = val_crop
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def prepare_data(self):
        self.train_recordings = [
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
        self.val_recordings = [
            "indoor_forward_10_davis_with_gt",
        ]

    def setup(self, stage):
        if stage == "fit":
            sequence = partial(
                FrameSequence,
                root_dir=self.root_dir,
                seq_len=self.train_seq_len,
                crop=self.train_crop,
                augmentations=self.augmentations,
            )
            recordings = []
            for rec in self.train_recordings:
                seq = sequence(recording=rec)
                recordings.extend([rec] * int(seq.n_frames / seq.seq_len))
            self.train_dataset = ConcatDataset([sequence(recording=rec) for rec in recordings])
            self.train_frame_shape = (self.batch_size, 3, *sequence(recording=recordings[0]).frame_shape)

        if stage in ["fit", "validate"]:
            sequence = partial(
                FrameSequence,
                root_dir=self.root_dir,
                crop=self.val_crop,
            )
            self.val_dataset = ConcatDataset([sequence(recording=rec) for rec in self.val_recordings])
            self.val_frame_shape = (1, 3, *sequence(recording=self.val_recordings[0]).frame_shape)

    def train_dataloader(self):
        sampler = ConcatBatchSampler(self.train_dataset, self.batch_size, shuffle=self.shuffle)
        dataloader = DataLoader(
            self.train_dataset, batch_sampler=sampler, num_workers=self.num_workers, collate_fn=time_first_collate
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=only_add_batch_dim,
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
