from bisect import bisect_left
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
from dotmap import DotMap
import h5py
import hdf5plugin
from lightning import LightningDataModule
import numpy as np
import numpy.lib.recfunctions as rfn
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import DataLoader, ConcatDataset

from .data_utils import batched, ConcatBatchSampler, InfiniteDataLoader, only_add_batch_dim, time_first_collate


@dataclass
class FlightSequence:
    root_dir: str
    recording: str
    time_window: int  # us
    chunk_size: int = 100
    drop_last_chunk: bool = False
    seq_len: int | None = None
    time: tuple[int, int] | None = None
    crop: tuple[int, ...] | None = None  # height, width or top, left, bottom, right
    rectify: bool = False
    augmentations: list[str] | None = None
    return_rotations: bool = False
    gt: list[str] | None = None

    def __post_init__(self):
        # defaults
        self.root_dir = Path(self.root_dir)

        # open large h5 files only once
        self.h5 = h5py.File(self.root_dir / f"{self.recording}.h5", "r")

        # forward rectification map
        # distorted -> rectified coords
        # provided map, easier than backward, but gives lines in frames due to nearest neighbor
        self.fw_rect_map = self.h5["fw_rect_map"][()]  # y_rect, x_rect = rect_map[y, x]

        # backward rectification map
        # rectified/undistorted -> distorted coords
        # more work, but prevents lines in accumulated event frames
        self.bw_rect_map = self.h5["bw_rect_map"][()]  # (h, w, 2)

        # other important attributes
        self.K_rect = self.h5.attrs["K_rect"]
        self.sensor_size = self.h5.attrs["sensor_size"]

        # get duration of recording
        # don't get full t because of memory usage
        self.t0, self.tk = self.h5["events/t"][[0, -1]]  # us
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

        # mapping from chunks to single steps
        # match seq_len if given
        self.chunk_size = self.seq_len if self.seq_len is not None else self.chunk_size
        self.chunk_map = batched(range(len(self.t_start)), self.chunk_size, drop_last=self.drop_last_chunk)

    def init_slice(self):
        if self.seq_len is not None:  # randomly-sliced sequence of seq_len
            t_rand = np.random.randint(
                self.t0, max(1, self.tk - (self.seq_len + 1) * self.time_window)
            )  # +1 for only full
            self.t_start = np.arange(t_rand, t_rand + self.seq_len * self.time_window, self.time_window)
            self.t_end = self.t_start + self.time_window
        else:  # full sequence
            self.t_start = np.arange(self.t0, self.tk - self.time_window, self.time_window)  # only full
            self.t_end = self.t_start + self.time_window

        self.seq_duration = self.t_end[-1] - self.t_start[0]

    def init_augmentation(self):
        self.augmentation = []
        if self.augmentations is not None:
            for aug in self.augmentations:
                if np.random.rand() < 0.5:
                    self.augmentation.append(aug)

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

    def reset(self):
        self.init_slice()  # slice up dataset
        self.init_crop()  # pre-compute crop
        self.init_augmentation()  # pre-compute augmentation

    def __len__(self):
        return len(self.chunk_map)

    def __getitem__(self, idx):
        # get new random slice, crop, augmentations
        self.reset()

        # get chunk
        chunk = self.chunk_map[idx]

        # go over slices
        events, frames, counts, rotations, targets = [], [], [], [], []
        for i in chunk:
            # convert to indices
            start = bisect_left(self.h5["events/t"], self.t_start[i])
            end = bisect_left(self.h5["events/t"], self.t_end[i])

            # get events as list
            t = self.h5["events/t"][start:end]  # uint32
            y = self.h5["events/y"][start:end]  # uint16
            x = self.h5["events/x"][start:end]  # uint16
            p = self.h5["events/p"][start:end]  # uint8 in {0, 1}

            # rectify events: forward rectification
            if self.rectify:
                x_rect, y_rect = self.fw_rect_map[y, x].T
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
            # use unrectified coordinates
            y = torch.from_numpy(y.astype(np.int64))
            x = torch.from_numpy(x.astype(np.int64))
            p = torch.from_numpy(p.astype(np.int64))
            frame = torch.zeros(2, *self.sensor_size, dtype=torch.int64)  # torch is faster
            frame.index_put_((p, y, x), torch.ones_like(p), accumulate=True)

            # rectify frame: backward rectification
            # backward to prevent lines in frames
            if self.rectify:
                frame = cv2.remap(frame.numpy().transpose(1, 2, 0), self.bw_rect_map, None, cv2.INTER_NEAREST)
                frame = torch.from_numpy(frame.transpose(2, 0, 1))

            # crop frame
            frame = frame[..., top:bottom, left:right]

            # discard if few events or same timestamp
            if len(lst) < 10 or lst["t"][-1] == lst["t"][0]:
                lst = np.array([], dtype=lst.dtype)
                frame = torch.zeros_like(frame)

            # format list of events: normalize time, polarity to {-1, 1}
            # after cropping, else normalized timestamp not correct
            lst["t"] = (lst["t"] - lst["t"][0]) / (lst["t"][-1] - lst["t"][0]) if len(lst) else lst["t"]
            lst["p"] = lst["p"] * 2 - 1

            # rotations
            # integrate angular vel to get rotation
            if self.return_rotations:
                start = bisect_left(self.h5["gt/imu_ts"], self.t_start[i])
                end = bisect_left(self.h5["gt/imu_ts"], self.t_end[i])
                if end - (start + 1) > 0:
                    rotation = R.identity()
                    for j in range(start + 1, end):
                        omega = self.h5["gt/imu_omega"][j]
                        omega *= np.array([-1, 1, -1])  # imu to camera frame
                        dt = (self.h5["gt/imu_ts"][j] - self.h5["gt/imu_ts"][j - 1]) * 1e-6  # us to s
                        rotation = rotation * R.from_rotvec(omega * dt)
                    rotation = rotation.as_rotvec()
                else:
                    rotation = R.identity().as_rotvec()
                rotations.append(rotation)

            # gt depth
            if self.gt and "depth" in self.gt:
                start = bisect_left(self.h5["gt/depth_ts"], self.t_start[i])
                end = bisect_left(self.h5["gt/depth_ts"], self.t_end[i])
                if end - start > 0:
                    try:
                        assert end - start == 1  # only one depth map per event_window
                    except AssertionError:
                        print(f"Multiple depth maps in event window {i}, taking latest")
                        start = end - 1
                    gt_depth_id = start + 1  # to prevent 0
                    gt_depth = self.h5["gt/depth"][start, 0] / 1000  # mm to m
                    # gt_depth = gt_depth[..., ::4, ::4]  # downsample 4x
                    gt_depth = self.depth_to_dvx(gt_depth, self.sensor_size, self.K_rect)
                    if self.crop is not None:
                        gt_depth = gt_depth[..., top:bottom, left:right]  # crop
                    # TODO: post processing?
                else:
                    gt_depth = None
                    gt_depth_id = None
            else:
                gt_depth = None
                gt_depth_id = None

            # gt color image
            if self.gt and "color" in self.gt:
                start = bisect_left(self.h5["gt/color_ts"], self.t_start[i])
                end = bisect_left(self.h5["gt/color_ts"], self.t_end[i])
                if end - start > 0:
                    try:
                        assert end - start == 1  # only one color frame per event_window
                    except AssertionError:
                        print(f"Multiple color frames in event window {i}, taking latest")
                        start = end - 1
                    gt_color = self.h5["gt/color"][start]
                else:
                    gt_color = None
            else:
                gt_color = None

            # append
            events.append(lst)
            frames.append(frame)
            counts.append(len(lst))
            targets.append(
                DotMap(
                    gt_depth=gt_depth,
                    gt_depth_id=gt_depth_id,
                    gt_color=gt_color,
                )
            )

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
        inv_K_rect = np.linalg.inv(K_rect)  # inv is fine here

        # convert to torch
        events = rfn.structured_to_unstructured(events, dtype=np.float32)
        events = torch.from_numpy(events)
        counts = torch.from_numpy(counts)
        auxs = DotMap(events=events, counts=counts)
        if self.return_rotations:
            rotations = torch.from_numpy(np.stack(rotations).astype(np.float32))
            auxs.gt_rotation = rotations
        targets = [
            DotMap(
                {k: torch.from_numpy(v.astype(np.float32)) if isinstance(v, np.ndarray) else v for k, v in t.items()}
            )
            for t in targets
        ]
        K_rect = torch.from_numpy(K_rect.astype(np.float32))
        inv_K_rect = torch.from_numpy(inv_K_rect.astype(np.float32))

        # return dotmap
        sample = DotMap()
        sample.frames = frames.float()
        sample.auxs = auxs
        if self.gt:
            sample.targets = targets
        sample.recording = self.recording
        sample.eofs = [i == len(self.t_start) - 1 for i in chunk]
        sample.K_rect = K_rect
        sample.inv_K_rect = inv_K_rect

        return sample

    @staticmethod
    def depth_to_dvx(depth_image, dvx_size, K_dvx):
        """
        Transform depth from Realsense (which is in RGB camera frame due to align_depth option) to event camera frame.
        """
        # get depth and event image dimensions
        depth_height, depth_width = depth_image.shape
        dvx_height, dvx_width = dvx_size

        # realsense rgb intrinsic matrix (because depth is aligned to rgb)
        K_depth = np.array(
            [[599.912109375, 0, 318.53460693359375], [0, 599.5509033203125, 247.19146728515625], [0, 0, 1]],
            dtype=np.float32,
        )
        # K_depth = np.array([[381.79150390625, 0, 322.4213562011719], [0, 381.79150390625, 234.92282104492188], [0, 0, 1]], dtype=np.float32)  # depth
        R = np.eye(3, dtype=np.float32)
        t = np.array([-0.033, 0.0275, -0.006], dtype=np.float32).reshape(3, 1)  # in meters

        # generate a grid of (u, v) pixel coordinates for the depth image
        u, v = np.meshgrid(np.arange(depth_width), np.arange(depth_height))
        ones = np.ones_like(u)

        # stack to create homogeneous coordinates (u, v, 1) for all pixels
        pixel_coords = np.stack([u, v, ones], axis=-1).reshape(-1, 3).T  # Shape: (3, depth_height*depth_width)

        # mask out invalid (zero) depth values
        depth_flattened = depth_image.flatten()
        valid_depth_mask = depth_flattened > 0
        depth_flattened = depth_flattened[valid_depth_mask]
        pixel_coords = pixel_coords[:, valid_depth_mask]  # Shape: (3, valid_points)

        # convert depth pixel coordinates to 3d points in depth camera space
        K_depth_inv = np.linalg.inv(K_depth)
        points_depth_camera = (K_depth_inv @ pixel_coords) * depth_flattened  # (3, valid_points)

        # transform points from depth camera to event camera
        points_dvx_camera = (R @ points_depth_camera) + t  # (3, valid_points)

        # project transformed 3d points onto the event camera image plane
        points_dvx_2d = K_dvx @ points_dvx_camera  # project 3d points in event camera to 2d
        points_dvx_2d = points_dvx_2d[:2] / points_dvx_2d[2]  # normalize by depth (homogeneous to 2d)

        # round and clip coordinates to fit within event image dimensions
        # TODO: adjust to event camera resolution?
        # u_rgb = np.round(points_rgb_2d[0] * (rgb_width / depth_width)).astype(int)
        # v_rgb = np.round(points_rgb_2d[1] * (rgb_height / depth_height)).astype(int)
        u_dvx = np.round(points_dvx_2d[0]).astype(int)
        v_dvx = np.round(points_dvx_2d[1]).astype(int)
        valid = (0 <= u_dvx) & (u_dvx < dvx_width) & (0 <= v_dvx) & (v_dvx < dvx_height)

        # transformed depth in event camera space
        # Z-buffering (keep the closest depth value)
        # transformed_depth = np.zeros((rgb_height, rgb_width), dtype=np.float32)
        # transformed_depth[v_rgb[valid], u_rgb[valid]] = points_rgb_camera[2, valid]
        transformed_depth = np.full((dvx_height, dvx_width), np.inf, dtype=np.float32)
        np.minimum.at(transformed_depth, (v_dvx[valid], u_dvx[valid]), points_dvx_camera[2, valid])
        transformed_depth[transformed_depth == np.inf] = 0

        return transformed_depth[None]


class FlightDataModule(LightningDataModule):
    gt = ["depth", "color"]

    def __init__(
        self,
        root_dir,
        time_window,
        train_seq_len,
        train_recordings,
        train_crop,
        val_recordings,
        val_crop,
        rectify,
        augmentations,
        return_rotations,
        return_events,
        batch_size,
        shuffle,
        num_workers,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.time_window = time_window
        self.train_seq_len = train_seq_len
        self.train_recordings = train_recordings
        self.train_crop = train_crop
        self.val_recordings = val_recordings
        self.val_crop = val_crop
        self.rectify = rectify
        self.augmentations = augmentations
        self.return_rotations = return_rotations
        self.return_events = return_events
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def prepare_data(self):
        # recordings
        # name, time (start, stop)
        # train: long sequence (8 minutes)
        # validate: short sequence (4 minutes)
        train_recordings = [("rosbag2_2024-11-01-12-05-04_0", (int(16e6), int(510e6)))]
        val_recordings = [("rosbag2_2024-11-01-11-13-10_0", (int(10e6), int(250e6)))]
        self.train_recordings = train_recordings if self.train_recordings is None else self.train_recordings
        self.val_recordings = val_recordings if self.val_recordings is None else self.val_recordings

    def setup(self, stage):
        if stage == "fit":
            train_sequence = partial(
                FlightSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                seq_len=self.train_seq_len,
                crop=self.train_crop,
                rectify=self.rectify,
                augmentations=self.augmentations,
                return_rotations=self.return_rotations,
            )
            train_recordings = []
            for rec, time in self.train_recordings:
                seq = train_sequence(recording=rec, time=time)
                train_recordings.extend([(rec, time)] * int(seq.rec_duration // seq.seq_duration))
            self.train_dataset = ConcatDataset(
                [train_sequence(recording=rec, time=time) for rec, time in train_recordings]
            )
            self.train_frame_shape = (self.batch_size, 2, *train_sequence(recording=train_recordings[0][0]).frame_shape)

        if stage in ["fit", "validate"]:
            val_sequence = partial(
                FlightSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                crop=self.val_crop,
                rectify=True,
                return_rotations=self.return_rotations,
                gt=self.gt,
            )
            self.val_dataset = ConcatDataset(
                [val_sequence(recording=rec, time=time) for rec, time in self.val_recordings]
            )
            self.val_frame_shape = (1, 2, *val_sequence(recording=self.val_recordings[0][0]).frame_shape)

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
