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

from depth_from_events.data_utils import (
    batched,
    ConcatBatchSampler,
    InfiniteDataLoader,
    only_add_batch_dim,
    time_first_collate,
)
from depth_from_events.dsec import DSEC_TRAIN_RECORDINGS, DSEC_VAL_RECORDINGS
from depth_from_events.uzh_fpv import UZH_FPV_TRAIN_RECORDINGS, UZH_FPV_VAL_RECORDINGS


@dataclass
class FrameSequence:
    root_dir: Path
    recording: str
    chunk_size: int = 100
    seq_len: int | None = None
    crop: tuple[int, ...] | None = None  # height, width or top, left, bottom, right
    augmentations: list[str] | None = None
    return_events: bool = False
    require_gt_poses: bool = False

    def __post_init__(self):
        # open large h5 files only once
        self.h5 = h5py.File(self.root_dir / f"{self.recording}.h5", "r")
        self.sensor_size = self.h5.attrs["sensor_size"]
        self.K_rect = torch.from_numpy(self.h5.attrs["K_rect"].astype(np.float32))

        # get number of frames
        self.n_frames = len(self.h5["events/frames"])

        # if require gt poses, restrict to frames with poses
        if "poses" in self.h5 and self.require_gt_poses:
            self.n_frames = self.h5["poses"].attrs["gt_pose_available_frames"]
            self.i_start_frame = self.h5["poses"].attrs["gt_pose_start_idx"]
            self.i_end_frame = self.h5["poses"].attrs["gt_pose_end_idx"]
        else:
            self.i_start_frame = 0
            self.i_end_frame = self.n_frames

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
            i_start = np.random.randint(self.i_start_frame, self.i_end_frame - self.seq_len)
            self.slice = list(range(i_start, i_start + self.seq_len))
        else:  # full sequence
            self.slice = list(range(self.i_start_frame, self.i_end_frame))

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
        start, stop = self.slice[chunk[0]], self.slice[chunk[-1]] + 1
        frames = torch.from_numpy(self.h5["events/frames"][start:stop].astype(np.float32))
        frames[:, 2:] *= self.h5.attrs["ts_res"]  # from int to quantized float again

        # if returning events, no need for avg ts channel
        if self.return_events:
            frames = frames[:, :2]

        # crop frames
        top, left, bottom, right = self.crop_corners
        frames = frames[..., top:bottom, left:right]

        # get pose associated with each frame
        if "poses" in self.h5:
            poses = torch.from_numpy(
                self.h5["poses"][start + self.i_start_frame : stop + self.i_start_frame].astype(np.float32)
            )
            translation, rotation = poses[:, :3], poses[:, 3:]
        else:
            translation, rotation = None, None

        # apply augmentations
        if "flip_t" in self.augmentation:
            frames = frames.flip(0)
            if not self.return_events:
                frames[:, 2:] = 1 - frames[:, 2:]  # revert avg ts in [0, 1]
            if translation is not None:
                translation = -translation
                rotation = -rotation
        if "flip_pol" in self.augmentation:
            frames[:, :2] = frames[:, :2].flip(1)  # only neg, pos
            if not self.return_events:
                frames[:, 2:] = frames[:, 2:].flip(1)  # flip avg ts
            if translation is not None:
                # nothing to do
                pass
        if "flip_ud" in self.augmentation:
            frames = frames.flip(2)
            if translation is not None:
                # flip ud -> translation reverse y axis
                translation[:, 1] = -translation[:, 1]
                # flip ud -> rotation reverse x, z axis
                rotation[:, 0] = -rotation[:, 0]
                rotation[:, 2] = -rotation[:, 2]
        if "flip_lr" in self.augmentation:
            frames = frames.flip(3)
            if translation is not None:
                # flip lr -> translation reverse x axis
                translation[:, 0] = -translation[:, 0]
                # flip lr -> rotation reverse y, z axis
                rotation[:, 1] = -rotation[:, 1]
                rotation[:, 2] = -rotation[:, 2]

        # get events
        if self.return_events:
            events, counts = [], []
            splits = self.h5["events/splits"][start:stop]
            if "flip_t" in self.augmentation:  # get windows in reverse
                splits = splits[::-1]
            for start, stop in splits:
                # get slice
                t = self.h5["events/t"][start:stop].astype(np.float64)  # float64
                y = self.h5["events/y"][start:stop]  # uint16 or float32
                x = self.h5["events/x"][start:stop]  # uint16 or float32
                p = self.h5["events/p"][start:stop].astype(np.float32)  # bool to float32

                # crop
                mask = (y >= top) & (y < bottom) & (x >= left) & (x < right)
                t, y, x, p = t[mask], y[mask], x[mask], p[mask]
                y, x = y - top, x - left  # rebase to crop

                # discard if roughly empty (like done with frames)
                if len(t) < 10 or t[-1] == t[0]:
                    events.append(np.zeros((0, 4), dtype=np.float32))
                    counts.append(0)
                    continue

                # formatting
                t_norm = (t - t[0]) / (t[-1] - t[0])  # normalize to [0, 1]
                p = p * 2 - 1  # to {-1, 1}

                # apply augmentations
                if "flip_t" in self.augmentation:
                    t_norm = 1 - t_norm  # revert ts in [0, 1]
                    t_norm, y, x, p = t_norm[::-1], y[::-1], x[::-1], p[::-1]  # make chronological
                if "flip_pol" in self.augmentation:
                    p *= -1
                if "flip_ud" in self.augmentation:
                    y = bottom - top - 1 - y
                if "flip_lr" in self.augmentation:
                    x = right - left - 1 - x

                events.append(np.stack([t_norm, y, x, p], axis=-1).astype(np.float32))
                counts.append(len(t))

            # pad sequences
            max_len = max(counts)
            events = np.stack([np.pad(ev, ((0, max_len - len(ev)), (0, 0))) for ev in events])
            events = torch.from_numpy(events)
            counts = torch.tensor(counts, dtype=torch.int64)
            auxs = DotMap(events=events, counts=counts)
        else:
            events = torch.zeros(self.chunk_size, 0, 4)
            counts = torch.zeros(self.chunk_size, dtype=torch.int64)
            auxs = DotMap(events=events, counts=counts)

        # adapt camera matrices to crop and augmentations
        K_rect = self.K_rect.clone()
        K_rect[0, 2] -= left
        K_rect[1, 2] -= top
        if "flip_ud" in self.augmentation:
            K_rect[1, 2] = (bottom - top - 1) - K_rect[1, 2]
        if "flip_lr" in self.augmentation:
            K_rect[0, 2] = (right - left - 1) - K_rect[0, 2]
        inv_K_rect = torch.linalg.inv(K_rect)

        # return dotmap
        sample = DotMap()
        sample.frames = frames
        sample.auxs = auxs
        sample.recording = self.recording
        sample.eofs = [i == len(self.slice) - 1 for i in chunk]
        sample.K_rect = K_rect
        sample.inv_K_rect = inv_K_rect
        if translation is not None:
            sample.pose = torch.cat([rotation, translation], dim=-1)

        return sample


class DataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        train_seq_len,
        train_crop,
        val_crop,
        augmentations,
        return_events,
        require_gt_poses,
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
        self.return_events = return_events
        self.require_gt_poses = require_gt_poses
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def setup(self, stage):
        if stage == "fit":
            sequence = partial(
                FrameSequence,
                root_dir=self.root_dir,
                seq_len=self.train_seq_len,
                crop=self.train_crop,
                augmentations=self.augmentations,
                return_events=self.return_events,
                require_gt_poses=self.require_gt_poses,
            )
            recordings = []
            for rec in self.train_recordings:
                seq = sequence(recording=rec)
                recordings.extend([rec] * int(seq.n_frames / (seq.seq_len if seq.seq_len else seq.n_frames)))
            self.train_dataset = ConcatDataset([sequence(recording=rec) for rec in recordings])
            channels = 2 if self.return_events else 4
            self.train_frame_shape = (self.batch_size, channels, *sequence(recording=recordings[0]).frame_shape)

        if stage in ["fit", "validate"]:
            sequence = partial(
                FrameSequence,
                root_dir=self.root_dir,
                crop=self.val_crop,
                return_events=self.return_events,
                require_gt_poses=self.require_gt_poses,
            )
            self.val_dataset = ConcatDataset([sequence(recording=rec) for rec in self.val_recordings])
            channels = 2 if self.return_events else 4
            self.val_frame_shape = (1, channels, *sequence(recording=self.val_recordings[0]).frame_shape)

    def train_dataloader(self):
        sampler = ConcatBatchSampler(self.train_dataset, self.batch_size, shuffle=self.shuffle)
        dataloader = InfiniteDataLoader(
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


class UzhFpvDataModule(DataModule):
    train_recordings = UZH_FPV_TRAIN_RECORDINGS
    val_recordings = UZH_FPV_VAL_RECORDINGS


class DsecDataModule(DataModule):
    train_recordings = DSEC_TRAIN_RECORDINGS
    val_recordings = DSEC_VAL_RECORDINGS


class FlightsDataModule(DataModule):
    train_recordings = ["rosbag2_2024-10-29-18-06-51_0"]
    val_recordings = ["rosbag2_2024-10-29-18-06-51_0"]


if __name__ == "__main__":
    from rich.progress import track

    datamodule = UzhFpvDataModule(
        root_dir="data/uzh_fpv_10ms_0.25ts_rect",
        train_seq_len=100,
        train_crop=(2, 1, 258, 345),
        val_crop=(2, 1, 258, 345),
        augmentations=["flip_t", "flip_pol", "flip_ud", "flip_lr"],
        return_events=True,
        batch_size=8,
        shuffle=True,
        num_workers=8,
    )
    datamodule = DsecDataModule(
        root_dir="data/dsec_10ms_0.25ts_rect",
        train_seq_len=100,
        train_crop=(128, 128),
        val_crop=None,
        augmentations=["flip_t", "flip_pol", "flip_ud", "flip_lr"],
        return_events=True,
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
