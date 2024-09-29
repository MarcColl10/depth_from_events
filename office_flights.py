from bisect import bisect_left
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from dotmap import DotMap
import h5py
import hdf5plugin
from lightning import LightningDataModule
import numpy as np
import numpy.lib.recfunctions as rfn
import torch
from torch.utils.data import DataLoader, ConcatDataset

from data_utils import batched, only_add_batch_dim


@dataclass
class FlightSequence:
    root_dir: str
    recording: str
    time_window: int  # us
    chunk_size: int = 100
    subsample: int | None = None

    def __post_init__(self):
        # defaults
        self.root_dir = Path(self.root_dir)
        self.sensor_size = (480, 640) if self.subsample is None else (480 // self.subsample, 640 // self.subsample)

        # open large h5 files only once
        self.h5 = h5py.File(self.root_dir / f"{self.recording}.h5", "r")

        # intrinsic matrix (fake)
        h, w = self.sensor_size
        fx, fy, cx, cy = [(h + w) / 2, (h + w) / 2, w / 2, h / 2]
        if self.subsample is not None:
            fx, fy, cx, cy = [v / self.subsample for v in [fx, fy, cx, cy]]
        self.K_rect = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.inv_K_rect = np.linalg.inv(self.K_rect)

        # get duration of recording
        # don't get full t because of memory usage
        self.t0, self.tk = self.h5["events/t"][[0, -1]]  # us
        self.rec_duration = self.tk - self.t0

        # slice dataset
        self.init_slice()

        # mapping from chunks to single steps
        self.chunk_map = batched(range(len(self.t_start)), self.chunk_size)

    def init_slice(self):
        self.t_start = np.arange(self.t0, self.tk - self.time_window, self.time_window)
        self.t_end = self.t_start + self.time_window

    def __len__(self):
        return len(self.chunk_map)

    def __getitem__(self, idx):
        # get chunk
        chunk = self.chunk_map[idx]

        # go over slices
        events, frames, counts = [], [], []
        for i in chunk:
            # convert to indices
            start = bisect_left(self.h5["events/t"], self.t_start[i])
            end = bisect_left(self.h5["events/t"], self.t_end[i])

            # get events as list
            t = self.h5["events/t"][start:end]  # uint32
            y = self.h5["events/y"][start:end]  # uint16
            x = self.h5["events/x"][start:end]  # uint16
            p = self.h5["events/p"][start:end]  # uint8 in {0, 1}

            # subsample
            if self.subsample is not None:
                y //= self.subsample
                x //= self.subsample

            # list of events to structured array
            dtype = np.dtype([("t", np.float64), ("y", np.float32), ("x", np.float32), ("p", np.int8)])
            lst = np.empty(len(t), dtype=dtype)
            lst["t"] = t
            lst["y"] = y
            lst["x"] = x
            lst["p"] = p

            # make into event count frame
            # use unrectified coordinates
            y = torch.from_numpy(y.astype(np.int64))
            x = torch.from_numpy(x.astype(np.int64))
            p = torch.from_numpy(p.astype(np.int64))
            frame = torch.zeros(2, *self.sensor_size, dtype=torch.int64)  # torch is faster
            frame.index_put_((p, y, x), torch.ones_like(p), accumulate=True)

            # discard if few events or same timestamp
            if len(lst) < 10:
                lst = np.array([], dtype=lst.dtype)
                frame = torch.zeros_like(frame)
            elif lst["t"][-1] == lst["t"][0]:
                lst = np.array([], dtype=lst.dtype)
                frame = torch.zeros_like(frame)

            # format list of events: normalize time, polarity to {-1, 1}
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

        # convert to torch
        events = rfn.structured_to_unstructured(events, dtype=np.float32)
        events = torch.from_numpy(events)
        counts = torch.from_numpy(counts)
        auxs = DotMap(events=events, counts=counts)
        K_rect = torch.from_numpy(self.K_rect)
        inv_K_rect = torch.from_numpy(self.inv_K_rect)

        # return dotmap
        sample = DotMap()
        sample.frames = frames.float()
        sample.auxs = auxs
        sample.recording = self.recording
        sample.eofs = [i == len(self.t_start) - 1 for i in chunk]
        sample.K_rect = K_rect
        sample.inv_K_rect = inv_K_rect

        return sample


class FlightDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        time_window,
        chunk_size,
        subsample,
        return_events,
        num_workers,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.time_window = time_window
        self.chunk_size = chunk_size
        self.subsample = subsample
        self.return_events = return_events
        self.num_workers = num_workers

    def prepare_data(self):
        # recordings
        # name, subsample
        recordings = [
            ("rosbag2_2024-09-19-14-06-54_0", None),
            ("rosbag2_2024-09-19-14-09-21_0", 2),
            ("rosbag2_2024-09-19-14-12-10_0", 4),
        ]
        self.recordings = [r for r, s in recordings if s == self.subsample]

    def setup(self, stage):
        sequence = partial(
            FlightSequence,
            root_dir=self.root_dir,
            time_window=self.time_window,
            chunk_size=self.chunk_size,
            subsample=self.subsample,
        )
        if stage == "fit":
            self.train_dataset = ConcatDataset([sequence(recording=rec) for rec in self.recordings])
            self.train_frame_shape = (1, 2, *sequence(recording=self.recordings[0]).sensor_size)
        if stage in ["fit", "validate"]:
            self.val_dataset = ConcatDataset([sequence(recording=rec) for rec in self.recordings])
            self.val_frame_shape = (1, 2, *sequence(recording=self.recordings[0]).sensor_size)

    def dataloader(self, stage):
        dataset = self.train_dataset if stage == "train" else self.val_dataset
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=only_add_batch_dim,
        )

    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("validate")


if __name__ == "__main__":
    from visualizer import RerunVisualizer

    visualizer = RerunVisualizer("office_flights")
    # sequence_full = FlightSequence("data/raw/office_flights", "rosbag2_2024-09-19-14-06-54_0", 10000)
    # sequence_half = FlightSequence("data/raw/office_flights", "rosbag2_2024-09-19-14-09-21_0", 10000)
    sequence_quarter = FlightSequence("data/raw/office_flights", "rosbag2_2024-09-19-14-12-10_0", 10000)

    for chunk in sequence_quarter:
        for frame in chunk.frames:
            visualizer.set_counter()
            visualizer.event_frame(frame)
