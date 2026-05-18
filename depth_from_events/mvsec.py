from bisect import bisect_left
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import zipfile

import cv2
from gdown import download_folder, download
import h5py
import hdf5plugin
from lightning import LightningDataModule
import numpy as np
from numpy.lib import recfunctions as rfn
import torch
from torch.utils.data import ConcatDataset, DataLoader
import yaml

from depth_from_events.data_utils import (
    batched,
    ConcatBatchSampler,
    InfiniteDataLoader,
    only_add_batch_dim,
    time_first_collate,
)


@dataclass
class MvsecSequence:
    root_dir: str
    recording: str
    time_window: float  # s
    chunk_size: int = 100
    seq_len: int | None = None
    time: tuple[float, float] | None = None  # start, end
    crop: tuple[int, ...] | None = None  # height, width or top, left, bottom, right
    rectify: bool = False
    augmentations: list[str] | None = None
    gt: list[str] | None = None

    def __post_init__(self):
        # defaults
        self.root_dir = Path(self.root_dir)
        self.sensor_size = (260, 346)  # height, width

        # make paths
        recording = self.recording[:-1]  # remove trailing number
        paths = {
            "data": self.root_dir / recording / f"{self.recording}_data.hdf5",
            "gt": self.root_dir / recording / f"{self.recording}_gt.hdf5",
            "rect_map_x": self.root_dir / recording / "calib" / f"{recording}_left_x_map.txt",
            "rect_map_y": self.root_dir / recording / "calib" / f"{recording}_left_y_map.txt",
            "calibration": self.root_dir / recording / "calib" / f"camchain-imucam-{recording}.yaml",
        }
        assert all(p.exists() for p in paths.values())

        # checks
        assert not (self.augmentations is not None and self.gt)  # no augmentations on gt

        # open large h5 files only once
        self.fs = dict(
            data=h5py.File(paths["data"], "r"),
            gt=h5py.File(paths["gt"], "r"),
        )

        # forward rectification map
        # distorted -> rectified coords
        # provided map, easier than backward, but gives lines in frames due to nearest neighbor
        rect_map_x = np.loadtxt(paths["rect_map_x"])
        rect_map_y = np.loadtxt(paths["rect_map_y"])
        self.fw_rect_map = np.stack([rect_map_x, rect_map_y], axis=-1)  # x_rect, y_rect = rect_map[y, x].T

        # backward rectification/undistortion map
        # rectified/undistorted -> distorted coords
        # more work, but prevents lines in accumulated event frames
        with open(paths["calibration"], "r") as f:
            cam_to_cam = yaml.safe_load(f)
        fx, fy, cx, cy = cam_to_cam["cam0"]["intrinsics"]
        K_dist = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.K_rect = np.array(cam_to_cam["cam0"]["projection_matrix"])[:, :3]
        R_rect = np.array(cam_to_cam["cam0"]["rectification_matrix"])
        dist_coeffs = np.array(cam_to_cam["cam0"]["distortion_coeffs"])
        resolution = cam_to_cam["cam0"]["resolution"]  # xy
        rect_map_x, rect_map_y = cv2.fisheye.initUndistortRectifyMap(
            K_dist, dist_coeffs, R_rect, self.K_rect, resolution, cv2.CV_32F
        )
        self.bw_rect_map = np.stack([rect_map_x, rect_map_y], axis=-1)  # needs to be .fisheye!

        # get duration of recording
        # don't get full t because of memory usage
        self.t0, self.tk = self.fs["data"]["davis/left/events"][[0, -1], 2]  # s
        if self.time is not None:
            t0, tk = self.time
            t0 = t0 + self.t0 if t0 is not None else self.t0
            tk = tk + self.t0 if tk is not None else self.tk
            self.t0, self.tk = t0, tk
        self.rec_duration = self.tk - self.t0

        # slice dataset, pre-compute crop and augmentation
        self.reset()

        # set frame shape
        self.frame_shape = (
            self.crop_corners[2] - self.crop_corners[0],
            self.crop_corners[3] - self.crop_corners[1],
        )

        # mapping from chunks to single slices
        # match seq_len if given
        self.chunk_size = self.seq_len if self.seq_len is not None else self.chunk_size
        self.chunk_map = batched(range(len(self.t_start)), self.chunk_size)

    def init_slice(self):
        # get start and end time
        t0, tk = self.t0, self.tk

        # randomize start time if seq_len
        if self.seq_len is not None:
            t_start = np.random.uniform(
                self.t0, max(self.t0, self.tk - (self.seq_len + 1) * self.time_window)
            )  # +1 for only full
            n_full_windows = self.seq_len
        else:
            t_start, t_end = t0, tk
            n_full_windows = max(1, int((t_end - t_start) // self.time_window))  # at least 1 window

        # window making
        # use linspace because floats (no rounding errors?)
        linspace = np.linspace(t_start, n_full_windows * self.time_window + t_start, n_full_windows + 1)
        self.t_start, self.t_end = linspace[:-1], linspace[1:]
        self.seq_duration = self.t_end[-1] - self.t_start[0]

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
        if self.augmentations is not None:
            for aug in self.augmentations:
                if np.random.rand() < 0.5:
                    self.augmentation.append(aug)

    def reset(self):
        self.init_slice()  # slice up dataset
        self.init_crop()  # pre-compute crop
        self.init_augmentation()  # pre-compute augmentation

    def __len__(self):
        return len(self.chunk_map)
    

    def get_gt_relative_pose(self, t_start, t_end):
        pose_ts = self.fs["gt"]["davis/left/pose_ts"]
        poses = self.fs["gt"]["davis/left/pose"]

        start_id = bisect_left(pose_ts, t_start)
        end_id = bisect_left(pose_ts, t_end)

        start_id = min(start_id, len(pose_ts) - 1)
        end_id = min(end_id, len(pose_ts) - 1)

        T0 = poses[start_id]
        T1 = poses[end_id]

        T_rel = np.linalg.inv(T0) @ T1

        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        rotvec, _ = cv2.Rodrigues(R_rel)
        rotvec = rotvec.reshape(3)

        pose = np.concatenate([rotvec, t_rel], axis=0).astype(np.float32)

        return pose


    def __getitem__(self, idx):
        # get new random slice, crop, augmentations
        self.reset()

        # get chunk
        chunk = self.chunk_map[idx]

        # go over slices
        events, frames, counts, targets, poses = [], [], [], [], []
        for i in chunk:
            # convert to indices
            start = bisect_left(self.fs["data"]["davis/left/events"], self.t_start[i], key=lambda x: x[2])
            end = bisect_left(self.fs["data"]["davis/left/events"], self.t_end[i], key=lambda x: x[2])

            # get events as list
            t = self.fs["data"]["davis/left/events"][start:end, 2]  # float64, s
            y = self.fs["data"]["davis/left/events"][start:end, 1]  # float64
            x = self.fs["data"]["davis/left/events"][start:end, 0]  # float64
            p = self.fs["data"]["davis/left/events"][start:end, 3]  # float64 in {-1, 1}

            # rectify list: forward rectification
            if self.rectify:
                x_rect, y_rect = self.fw_rect_map[y.astype(np.int64), x.astype(np.int64)].T
            else:
                x_rect, y_rect = x, y

            # list of events to structured array
            dtype = np.dtype([("t", np.float64), ("y", np.float32), ("x", np.float32), ("p", np.int8)])
            lst = np.empty(len(t), dtype=dtype)
            lst["t"] = t
            lst["y"] = y_rect
            lst["x"] = x_rect
            lst["p"] = p

            # crop list
            top, left, bottom, right = self.crop_corners
            mask = (y_rect >= top) & (y_rect < bottom) & (x_rect >= left) & (x_rect < right)
            lst = lst[mask]
            lst["y"] -= top
            lst["x"] -= left

            # make into event count frame
            # use unrectified coordinates, convert p to {0, 1}
            y = torch.from_numpy(y.astype(np.int64))
            x = torch.from_numpy(x.astype(np.int64))
            p = torch.from_numpy(((p + 1) // 2).astype(np.int64))
            frame = torch.zeros(2, *self.sensor_size, dtype=torch.int64)  # torch is faster
            frame.index_put_((p, y, x), torch.ones_like(p), accumulate=True)

            # rectify frame: backward rectification
            # backward to prevent lines in frames
            if self.rectify:
                frame = cv2.remap(
                    frame.numpy().transpose(1, 2, 0), self.bw_rect_map, None, interpolation=cv2.INTER_NEAREST
                )
                frame = torch.from_numpy(frame.transpose(2, 0, 1))

            # crop frame
            frame = frame[..., top:bottom, left:right]

            # discard if few events or same timestamp
            if len(lst) < 10 or lst["t"][-1] == lst["t"][0]:
                lst = np.array([], dtype=lst.dtype)
                frame = torch.zeros_like(frame)

            # format list of events
            # after cropping, else normalized timestamp not correct
            # only normalize time; polarity is already in {-1, 1}
            lst["t"] = (lst["t"] - lst["t"][0]) / (lst["t"][-1] - lst["t"][0]) if len(lst) else lst["t"]

            # gt depth
            if self.gt and "depth" in self.gt:
                start = bisect_left(self.fs["gt"]["davis/left/depth_image_rect_ts"], self.t_start[i])
                end = bisect_left(self.fs["gt"]["davis/left/depth_image_rect_ts"], self.t_end[i])
                if end - start > 0:
                    if end - start > 1:
                        print(f"Multiple depth maps in event window {i}, taking latest")
                        start = end - 1
                    gt_depth_id = start + 1  # to prevent 0
                    gt_depth = self.fs["gt"]["davis/left/depth_image_rect"][start:end]  # keep time dim as channel
                    gt_depth[..., 192:, :] = 0  # remove car bonnet for car recordings
                    gt_depth = gt_depth[..., top:bottom, left:right]  # crop
                    gt_depth[np.isnan(gt_depth)] = 0  # replace nans with 0
                else:
                    gt_depth = None
                    gt_depth_id = None
            else:
                gt_depth = None
                gt_depth_id = None

            # append
            events.append(lst)
            frames.append(frame)
            counts.append(len(lst))
            targets.append(dict(gt_depth=gt_depth, gt_depth_id=gt_depth_id))
            poses.append(self.get_gt_relative_pose(self.t_start[i], self.t_end[i]))

        # stack and pad
        max_len = max(counts)
        events = [np.pad(ev, (0, max_len - len(ev))) for ev in events]
        events = np.stack(events)
        frames = torch.stack(frames)
        counts = np.array(counts)

        # apply augmentations; more efficient on chunks
        # not used with targets, so leave those out
        if "backward" in self.augmentation:
            events["t"] = 1 - events["t"]
            events = np.flip(events)
            frames = frames.flip(0)
            counts = np.flip(counts).copy()
        if "vertical" in self.augmentation:
            events["y"] = (bottom - top - 1) - events["y"]
            frames = frames.flip(-2)
        if "horizontal" in self.augmentation:
            events["x"] = (right - left - 1) - events["x"]
            frames = frames.flip(-1)
        if "polarity" in self.augmentation:
            events["p"] *= -1
            frames = frames.flip(-3)  # only flip polarity

        # adapt camera matrices to crop and augmentations
        K_rect = self.K_rect.copy()
        K_rect[0, 2] -= left
        K_rect[1, 2] -= top
        if "vertical" in self.augmentation:
            K_rect[1, 2] = (bottom - top - 1) - K_rect[1, 2]
        if "horizontal" in self.augmentation:
            K_rect[0, 2] = (right - left - 1) - K_rect[0, 2]
        inv_K_rect = np.linalg.inv(K_rect)

        # convert to torch
        events = rfn.structured_to_unstructured(events, dtype=np.float32)
        events = torch.from_numpy(events)
        counts = torch.from_numpy(counts)
        poses = torch.from_numpy(np.stack(poses).astype(np.float32))

        auxs = dict(events=events, counts=counts)
        targets = [
            {k: torch.from_numpy(v.astype(np.float32)) if isinstance(v, np.ndarray) else v for k, v in t.items()}
            for t in targets
        ]
        K_rect = torch.from_numpy(K_rect.astype(np.float32))
        inv_K_rect = torch.from_numpy(inv_K_rect.astype(np.float32))

        # return dict
        sample = dict(
            frames=frames.float(),
            poses=poses,
            auxs=auxs,
            targets=targets if self.gt else None,
            recording=self.recording,
            eofs=[i == len(self.t_start) - 1 for i in chunk],
            K_rect=K_rect,
            inv_K_rect=inv_K_rect,
        )

        return sample


class MvsecDataModule(LightningDataModule):
    gt = ["depth"]

    def __init__(
        self,
        root_dir,
        time_window,
        train_seq_len,
        train_recordings,
        train_time,
        train_crop,
        val_recordings,
        val_time,
        val_crop,
        rectify,
        augmentations,
        return_events,
        batch_size,
        shuffle,
        num_workers,
        download,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.time_window = time_window
        self.train_seq_len = train_seq_len
        self.train_recordings = train_recordings
        self.train_time = train_time
        self.train_crop = train_crop
        self.val_recordings = val_recordings
        self.val_time = val_time
        self.val_crop = val_crop
        self.rectify = rectify
        self.augmentations = augmentations
        self.return_events = return_events
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.download = download

    def prepare_data(self):
        if self.download:
            # event and gt data in h5
            # parent: https://drive.google.com/drive/folders/1gDy2PwVOu_FPOsEZjojdWEB2ZHmpio8D
            urls = [
                ("indoor_flying", "https://drive.google.com/drive/folders/1CEuvvahWQntNIqXWZhXu_WknsTLm4Sum"),
                ("outdoor_day", "https://drive.google.com/drive/folders/1WUapfrd2DNQNuxPt9IqUHCcPCPKLiNvT"),
            ]

            # download if not there
            for name, url in urls:
                (self.root_dir / name).mkdir(exist_ok=True, parents=True)
                files = download_folder(url, output=str(self.root_dir / name), skip_download=True)
                for f in files:
                    id, _, local_path = f
                    if not Path(local_path).exists():
                        try:
                            download(id=id, output=local_path)
                        except Exception as e:
                            print(e)
                    if ".zip" in local_path:
                        if not (self.root_dir / name / "calib").exists():
                            with zipfile.ZipFile(local_path, "r") as f:
                                f.extractall(self.root_dir / name / "calib")
                        (self.root_dir / name / f"{name}_calib.zip").unlink()

        # train on outdoor_day2, validate on part of outdoor_day1 (default)
        train_recordings = ["outdoor_day2"] if self.train_recordings is None else self.train_recordings  # override
        train_time = [(0, None)] if self.train_time is None else self.train_time
        val_recordings = ["outdoor_day1"] if self.val_recordings is None else self.val_recordings
        val_time = [(222.4, 240.4)] if self.val_recordings is None else self.val_time

        # store for building datasets later
        self.train_recordings = train_recordings
        self.train_time = train_time
        self.val_recordings = val_recordings
        self.val_time = val_time

    def setup(self, stage):
        if stage == "fit":
            train_sequence = partial(
                MvsecSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                seq_len=self.train_seq_len,
                crop=self.train_crop,
                rectify=self.rectify,
                augmentations=self.augmentations,
            )
            train_recordings, train_time = [], []
            for rec, time in zip(self.train_recordings, self.train_time):
                seq = train_sequence(recording=rec, time=time)
                train_recordings.extend([rec] * int(seq.rec_duration // seq.seq_duration))
                train_time.extend([time] * int(seq.rec_duration // seq.seq_duration))
            self.train_dataset = ConcatDataset(
                [train_sequence(recording=rec, time=time) for rec, time in zip(train_recordings, train_time)]
            )
            self.train_frame_shape = (self.batch_size, 2, *train_sequence(recording=train_recordings[0]).frame_shape)

        if stage in ["fit", "validate"]:
            val_sequence = partial(
                MvsecSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                crop=self.val_crop,
                rectify=True,
                gt=self.gt,
            )
            self.val_dataset = ConcatDataset(
                [val_sequence(recording=rec, time=time) for rec, time in zip(self.val_recordings, self.val_time)]
            )
            self.val_frame_shape = (1, 2, *val_sequence(recording=self.val_recordings[0]).frame_shape)

        elif stage == "test":
            raise NotImplementedError

    def train_dataloader(self):
        sampler = ConcatBatchSampler(self.train_dataset, self.batch_size, shuffle=self.shuffle)
        return InfiniteDataLoader(
            self.train_dataset, batch_sampler=sampler, num_workers=self.num_workers, collate_fn=time_first_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers // 2,
            collate_fn=only_add_batch_dim,
        )


if __name__ == "__main__":
    datamodule = MvsecDataModule(
        root_dir="data/raw/mvsec",
        time_window=0.01,  # s
        train_seq_len=100,
        train_crop=None,
        val_recordings=None,
        val_time=None,
        val_crop=None,
        rectify=True,
        augmentations=["backward", "polarity", "horizontal"],
        batch_size=8,
        shuffle=True,
        num_workers=8,
        download=True,
    )
    datamodule.prepare_data()
