from bisect import bisect_left
from pathlib import Path
import tempfile

import cv2
import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from rich.progress import track
from torchvision.datasets.utils import download_and_extract_archive
import yaml


def get_uzh_fpv_h5_frames(root_dir, time_window, count_window, ts_res, rectify):
    # now only indoor forward, but there's also 45deg and outdoor
    # time in seconds to skip at the beginning to approx. start at takeoff
    root_dir = Path(root_dir)
    sensor_size = (260, 346)  # height, width
    recordings = {
        "indoor_forward_3_davis_with_gt": 30.0,
        "indoor_forward_5_davis_with_gt": 30.0,
        "indoor_forward_6_davis_with_gt": 30.0,
        "indoor_forward_7_davis_with_gt": 30.0,
        "indoor_forward_8_davis": 30.0,
        "indoor_forward_9_davis_with_gt": 30.0,
        "indoor_forward_10_davis_with_gt": 30.0,
        "indoor_forward_11_davis": 30.0,
        "indoor_forward_12_davis": 20.0,
    }

    # download
    base_url_rec = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/"
    base_url_calib = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv/calib/"

    for rec, t0_skip in recordings.items():
        name = ("_").join(rec.split("_")[:2])  # eg indoor_forward

        dest = root_dir / name
        dest.mkdir(parents=True, exist_ok=True)

        # calibration
        if not (dest / "calib.yaml").exists():  # calibration
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)
                download_and_extract_archive(f"{base_url_calib}{name}_calib_davis.zip", tmp_dir)
                (tmp_dir / f"{name}_calib_davis" / f"camchain-..{name}_calib_davis_cam.yaml").rename(
                    dest / "calib.yaml"
                )

        # recording
        if not (dest / f"{rec}.h5").exists():
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)
                download_and_extract_archive(f"{base_url_rec}{rec}.zip", tmp_dir)

                def append(dataset, data):
                    n = len(data)
                    dataset.resize(len(dataset) + n, axis=0)
                    dataset[-n:] = data

                # first get raw data so we can work with it efficiently
                # then make frames with channels: neg count, pos count, avg quantized ts
                with h5py.File(dest / f"{rec}.h5", "w") as h5f:
                    h5f.create_dataset(
                        "events/frames",
                        (0, 3, *sensor_size),
                        maxshape=(None, 3, *sensor_size),
                        chunks=True,
                        dtype=np.float32,
                        **hdf5plugin.Zstd(),
                    )
                    h5f.create_dataset(
                        "events/t", (0,), maxshape=(None,), chunks=True, dtype=np.float64, **hdf5plugin.Zstd()
                    )
                    h5f.create_dataset(
                        "events/y", (0,), maxshape=(None,), chunks=True, dtype=np.uint16, **hdf5plugin.Zstd()
                    )
                    h5f.create_dataset(
                        "events/x", (0,), maxshape=(None,), chunks=True, dtype=np.uint16, **hdf5plugin.Zstd()
                    )
                    h5f.create_dataset(
                        "events/p", (0,), maxshape=(None,), chunks=True, dtype=np.bool_, **hdf5plugin.Zstd()
                    )

                    events = pd.read_csv(
                        tmp_dir / "events.txt",
                        delimiter=" ",
                        skiprows=1,
                        names=["t", "x", "y", "p"],
                        chunksize=1e6,
                    )

                    # put in raw
                    for df in track(events, description=f"Converting {rec} to h5..."):
                        append(h5f["events/t"], df["t"].values)
                        append(h5f["events/y"], df["y"].values)
                        append(h5f["events/x"], df["x"].values)
                        append(h5f["events/p"], df["p"].values)

                    # proces into frames
                    if time_window is not None:
                        t0, tk = h5f["events/t"][0], h5f["events/t"][-1]
                        t0 += t0_skip
                        n_full_windows = int((tk - t0) // time_window)
                        t_split = np.linspace(t0, tk, n_full_windows + 1)
                        splits = np.searchsorted(h5f["events/t"], t_split)
                    elif count_window is not None:
                        start = bisect_left(h5f["events/t"], h5f["events/t"][0] + t0_skip)
                        splits = np.arange(start, len(h5f["events/t"]), count_window)

                    # precompute backwards rectification
                    if rectify:
                        # kalibr equidistant = .fisheye
                        with open(dest / "calib.yaml", "r") as f:
                            cam_to_cam = yaml.safe_load(f)
                        fx, fy, cx, cy = cam_to_cam["cam0"]["intrinsics"]
                        K_dist = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                        K_rect = K_dist.copy()  # usually same for fisheye
                        dist_coeffs = np.array(cam_to_cam["cam0"]["distortion_coeffs"])
                        resolution = cam_to_cam["cam0"]["resolution"]  # xy
                        rect_map_x, rect_map_y = cv2.fisheye.initUndistortRectifyMap(
                            K_dist, dist_coeffs, np.eye(3), K_rect, resolution, cv2.CV_32F
                        )
                        bw_rect_map = np.stack([rect_map_x, rect_map_y], axis=-1)

                    chunk_size = 100
                    chunks = np.array_split(np.stack([splits[:-1], splits[1:]]), len(splits) // chunk_size, axis=1)
                    for chunk in track(chunks, description=f"Converting {rec} to frames..."):
                        starts, stops = chunk
                        frames = []
                        for start, stop in zip(starts, stops):
                            t = h5f["events/t"][start:stop]  # float64
                            y = h5f["events/y"][start:stop]  # uint16
                            x = h5f["events/x"][start:stop]  # uint16
                            p = h5f["events/p"][start:stop].astype(np.uint8)  # bool to uint8

                            # discard if few events or same timestamp
                            if len(t) < 10 or t[-1] == t[0]:
                                frame = np.zeros((3, *sensor_size), dtype=np.float32)
                                frames.append(frame)
                                continue

                            # normalize timestamp
                            t_norm = (t - t[0]) / (t[-1] - t[0])

                            # make into event count frame
                            # channels neg, pos, avg ts (quantized)
                            frame = np.zeros((3, *sensor_size), dtype=np.float32)
                            np.add.at(frame[:2], (p, y, x), 1)
                            np.add.at(frame[-1], (y, x), t_norm)
                            frame[-1] = np.round(frame[-1] / (frame[:2].sum(0) + 1e-9) / ts_res) * ts_res

                            # backwards rectification
                            if rectify:
                                frame = cv2.remap(frame.transpose(1, 2, 0), bw_rect_map, None, cv2.INTER_NEAREST)
                                frame = frame.transpose(2, 0, 1)

                            frames.append(frame)

                        # add to h5
                        frames = np.stack(frames)
                        append(h5f["events/frames"], frames)


if __name__ == "__main__":
    get_uzh_fpv_h5_frames(
        "data/uzh_fpv_10ms_0.25ts_rect", time_window=0.01, count_window=None, ts_res=0.25, rectify=True
    )

    import rerun as rr

    rr.init("uzh_fpv_test")
    rr.serve()

    with h5py.File("data/uzh_fpv_10ms_0.25ts_rect/indoor_forward/indoor_forward_12_davis.h5", "r") as f:
        frames = f["events/frames"]
        for frame in frames:
            rr.log("frame", rr.Tensor(frame, dim_names=["C", "H", "W"]))
