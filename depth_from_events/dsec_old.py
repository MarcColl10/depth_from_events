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
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets.utils import download_and_extract_archive, download_url
import yaml

from .data_utils import batched, ConcatBatchSampler, InfiniteDataLoader, only_add_batch_dim, time_first_collate


@dataclass
class DsecSequence:
    root_dir: str
    recording: str
    time_window: int  # us
    count_thresh: int | None = None
    chunk_size: int = 100
    seq_len: int | None = None
    crop: tuple[int, ...] | None = None  # height, width or top, left, bottom, right
    rectify: bool = False
    augmentations: list[str] | None = None

    def __post_init__(self):
        # defaults
        self.root_dir = Path(self.root_dir)
        self.sensor_size = (480, 640)  # height, width

        # make paths
        paths = {
            "data": self.root_dir / self.recording / "events_left" / "events.h5",
            "rectify_map": self.root_dir / self.recording / "events_left" / "rectify_map.h5",
            "calibration": self.root_dir / self.recording / "calibration" / "cam_to_cam.yaml",
        }
        assert all(p.exists() for p in paths.values())

        # checks
        assert not (self.count_thresh is not None and self.seq_len is None)  # count_thresh changes chunk mapping

        # open large h5 files only once
        self.fs = DotMap()
        self.fs.data = h5py.File(paths["data"], "r")

        # store other paths for later
        self.paths = paths

        # forward rectification map
        # only load/make rectification maps once
        # distorted -> rectified coords
        # provided map, easier than backward, but gives lines in frames due to nearest neighbor
        with h5py.File(paths["rectify_map"], "r") as f:
            self.fw_rect_map = f["rectify_map"][()]  # x_rect, y_rect = rect_map[y, x].T

        # backward rectification/undistortion map
        # rectified/undistorted -> distorted coords
        # more work, but prevents lines in accumulated event frames
        with open(paths["calibration"], "r") as f:
            cam_to_cam = yaml.safe_load(f)
        fx, fy, cx, cy = cam_to_cam["intrinsics"]["cam0"]["camera_matrix"]  # distorted image
        K_dist = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        fx, fy, cx, cy = cam_to_cam["intrinsics"]["camRect0"]["camera_matrix"]  # rectified image
        self.K_rect = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        R_rect = np.array(cam_to_cam["extrinsics"]["R_rect0"])
        dist_coeffs = np.array(cam_to_cam["intrinsics"]["cam0"]["distortion_coeffs"])
        resolution = cam_to_cam["intrinsics"]["cam0"]["resolution"]  # xy
        self.bw_rect_map, _ = cv2.initUndistortRectifyMap(
            K_dist, dist_coeffs, R_rect, self.K_rect, resolution, cv2.CV_32FC2
        )

        # get duration of recording
        # don't get full t because of memory usage
        self.t0, self.tk = self.fs.data["events/t"][[0, -1]] + self.fs.data["t_offset"][()]  # us
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
        self.chunk_map = batched(range(len(self.t_start)), self.chunk_size)

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

    def __getitem__(self, idx):
        # get new random slice, crop, augmentations
        self.reset()

        # get chunk
        chunk = self.chunk_map[idx]

        # go over slices
        events, frames, counts = [], [], []
        for i in chunk:
            # convert to indices
            offset = self.fs.data["t_offset"][()]
            start = bisect_left(self.fs.data["events/t"], self.t_start[i] - offset)
            end = bisect_left(self.fs.data["events/t"], self.t_end[i] - offset)

            # get events as list
            t = self.fs.data["events/t"][start:end]  # uint32
            y = self.fs.data["events/y"][start:end]  # uint16
            x = self.fs.data["events/x"][start:end]  # uint16
            p = self.fs.data["events/p"][start:end]  # uint8 in {0, 1}

            # rectify list: forward rectification
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

            # compute crop for list
            top, left, bottom, right = self.crop_corners
            mask = (y_rect >= top) & (y_rect < bottom) & (x_rect >= left) & (x_rect < right)

            # if event count above threshold: shorten window
            # TODO: when rectifying, this will give wrong sum of events in image
            # because crop_mask is based on rectified while image uses unrectified
            if self.count_thresh is not None and mask.sum() > self.count_thresh:
                cumsum = np.cumsum(mask)
                new_end = bisect_left(cumsum, self.count_thresh) + 1

                lst = lst[:new_end]
                mask = mask[:new_end]
                y = y[:new_end]
                x = x[:new_end]
                p = p[:new_end]

                new_t_end = self.fs.data["events/t"][start + new_end] + offset
                dt = self.t_end[i] - new_t_end
                self.t_end[i:] -= dt
                if i < len(self.t_start) - 1:
                    self.t_start[i + 1 :] -= dt

            # crop list
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
            # because https://github.com/uzh-rpg/DSEC/issues/16 (lines in frames)
            # following https://github.com/uzh-rpg/DSEC/issues/14#issuecomment-841348958
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

            # format list of events: normalize time, polarity to {-1, 1}
            # after cropping, else normalized timestamp not correct
            lst["t"] = (lst["t"] - lst["t"][0]) / (lst["t"][-1] - lst["t"][0]) if len(lst) else lst["t"]
            lst["p"] = lst["p"] * 2 - 1

            # append
            events.append(lst)
            frames.append(frame)
            counts.append(len(lst))

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

        # adapt camera matrices to crop, resize and augmentations
        K_rect = self.K_rect.copy()
        K_rect[0, 2] -= left
        K_rect[1, 2] -= top
        if "vertical" in self.augmentation:
            K_rect[1, 2] = (bottom - top - 1) - K_rect[1, 2]
        if "horizontal" in self.augmentation:
            K_rect[0, 2] = (right - left - 1) - K_rect[0, 2]
        inv_K_rect = np.linalg.pinv(K_rect)

        # convert to torch
        events = rfn.structured_to_unstructured(events, dtype=np.float32)
        events = torch.from_numpy(events)
        counts = torch.from_numpy(counts)
        auxs = DotMap(events=events, counts=counts)
        K_rect = torch.from_numpy(K_rect.astype(np.float32))
        inv_K_rect = torch.from_numpy(inv_K_rect.astype(np.float32))

        # return dotmap
        sample = DotMap()
        sample.frames = frames.float()
        sample.auxs = auxs
        sample.recording = self.recording
        sample.eofs = [i == len(self.t_start) - 1 for i in chunk]
        sample.K_rect = K_rect
        sample.inv_K_rect = inv_K_rect

        return sample


class DsecDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        time_window,
        count_thresh,
        train_seq_len,
        train_crop,
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
        self.count_thresh = count_thresh
        self.train_seq_len = train_seq_len
        self.train_crop = train_crop
        self.val_crop = val_crop
        self.rectify = rectify
        self.augmentations = augmentations
        self.return_events = return_events
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.download = download

    def prepare_data(self):
        # all train recordings excluding night
        # boolean indicates flow gt available (all have disparity gt)
        train_recordings = {
            "interlaken_00_c": False,
            "interlaken_00_d": False,
            "interlaken_00_e": False,
            "interlaken_00_f": False,
            "interlaken_00_g": False,
            # "thun_00_a": True,
            # "zurich_city_01_a": True,
            "zurich_city_04_a": False,
            "zurich_city_04_b": False,
            "zurich_city_04_c": False,
            "zurich_city_04_d": False,
            "zurich_city_04_e": False,
            "zurich_city_04_f": False,
            "zurich_city_05_a": True,
            "zurich_city_05_b": True,
            "zurich_city_06_a": True,
            "zurich_city_07_a": True,
            "zurich_city_08_a": True,
            # "zurich_city_09_a": True,
            "zurich_city_11_a": True,
            "zurich_city_11_b": True,
            "zurich_city_11_c": True,
        }
        # some from training set as validation
        val_recordings = {
            "thun_00_a": True,  # daylight
            "zurich_city_01_a": True,  # darkish
            "zurich_city_09_a": True,  # night
        }
        train_val_recordings = {**train_recordings, **val_recordings}

        # all test recordings
        # boolean indicates whether in the flow benchmark (all are in disparity benchmark)
        test_recordings = {
            "interlaken_00_a": False,
            "interlaken_00_b": True,
            "interlaken_01_a": True,
            "thun_01_a": True,
            "thun_01_b": True,
            "zurich_city_12_a": True,
            "zurich_city_13_a": False,
            "zurich_city_13_b": False,
            "zurich_city_14_a": False,
            "zurich_city_14_b": False,
            "zurich_city_14_c": True,
            "zurich_city_15_a": True,
        }

        # download all recordings
        if self.download:
            base_url = "https://download.ifi.uzh.ch/rpg/DSEC/"
            data_names = {
                "events_left": ".zip",
                "calibration": ".zip",
            }
            target_names = {
                "disparity_event": ".zip",
                "disparity_timestamps": ".txt",
                "optical_flow_forward_event": ".zip",
                "optical_flow_forward_timestamps": ".txt",
            }

            # training recordings
            for rec, has_flow in train_val_recordings.items():
                dest = self.root_dir / rec
                dest.mkdir(parents=True, exist_ok=True)

                for name, ext in {**data_names, **target_names}.items():
                    if "flow" in name and not has_flow:
                        continue
                    else:
                        if not (dest / name).exists():
                            if ext == ".zip":
                                print(f"{base_url}/train/{rec}/{rec}_{name}{ext}", dest)
                                download_and_extract_archive(f"{base_url}/train/{rec}/{rec}_{name}{ext}", dest / name)
                                (dest / name / f"{rec}_{name}{ext}").unlink()
                            else:
                                download_url(f"{base_url}/train/{rec}/{rec}_{name}{ext}", dest / name)

            # test recordings
            for rec in test_recordings:
                dest = self.root_dir / rec
                dest.mkdir(parents=True, exist_ok=True)

                for name, ext in data_names.items():
                    if not (dest / name).exists():
                        if ext == ".zip":
                            download_and_extract_archive(f"{base_url}/test/{rec}/{rec}_{name}{ext}", dest / name)
                            (dest / name / f"{rec}_{name}{ext}").unlink()
                        else:
                            download_url(f"{base_url}/test/{rec}/{rec}_{name}{ext}", dest / name)

            # flow and disparity evaluation timestamps
            eval_names = {
                "flow_eval_timestamps": "test_forward_optical_flow_timestamps.zip",
                "disparity_eval_timestamps": "test_disparity_timestamps.zip",
            }
            for rename, name in eval_names.items():
                if not len(list(self.root_dir.rglob(rename))):
                    download_and_extract_archive(f"{base_url}{name}", self.root_dir)
                    (self.root_dir / f"{name}").unlink()
                    for rec, in_flow in test_recordings.items():
                        if "flow" in name and not in_flow:
                            continue
                        else:
                            dest = self.root_dir / rec / rename
                            dest.mkdir(parents=True, exist_ok=True)
                            (self.root_dir / f"{rec}.csv").rename(dest / f"{rec}.csv")

        # store for building datasets later
        self.train_recordings = train_recordings
        self.val_recordings = val_recordings
        self.test_recordings = test_recordings

    def setup(self, stage):
        if stage == "fit":
            train_sequence = partial(
                DsecSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                count_thresh=self.count_thresh,
                seq_len=self.train_seq_len,
                crop=self.train_crop,
                rectify=self.rectify,
                augmentations=self.augmentations,
            )
            train_recordings = []
            for rec in self.train_recordings:
                seq = train_sequence(recording=rec)
                train_recordings.extend([rec] * int(seq.rec_duration / seq.seq_duration))
            self.train_dataset = ConcatDataset([train_sequence(recording=rec) for rec in train_recordings])
            self.train_frame_shape = (self.batch_size, 2, *train_sequence(recording=train_recordings[0]).frame_shape)

        if stage in ["fit", "validate"]:
            val_sequence = partial(
                DsecSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                crop=self.val_crop,
                rectify=True,
            )
            self.val_dataset = ConcatDataset([val_sequence(recording=rec) for rec in self.val_recordings])
            self.val_frame_shape = (
                self.batch_size,
                2,
                *val_sequence(recording=list(self.val_recordings.keys())[0]).frame_shape,
            )

        elif stage == "test":
            test_sequence = partial(
                DsecSequence,
                root_dir=self.root_dir,
                time_window=self.time_window,
                rectify=True,
            )
            self.test_dataset = ConcatDataset([test_sequence(recording=rec) for rec in self.test_recordings])
            self.test_frame_shape = (
                self.batch_size,
                2,
                *test_sequence(recording=list(self.test_recordings.keys())[0]).frame_shape,
            )

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

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers // 2,
            collate_fn=only_add_batch_dim,
        )
