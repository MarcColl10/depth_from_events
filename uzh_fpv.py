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
    """
    Convert UZH-FPV dataset to h5 files containing raw events and event frames.

    Args:
        root_dir (str): Root directory to save the dataset.
        time_window (float): Time window in seconds to split the dataset into frames.
        count_window (int): Number of events to split the dataset into frames.
        ts_res (float): Timestamp resolution to quantize the average timestamp channel.
        rectify (bool): Whether to rectify the frames using the calibration data.
    """

    # now only indoor forward, but there's also 45deg and outdoor
    # time in seconds to skip at the beginning to approx. start at takeoff
    root_dir = Path(root_dir)
    sensor_size = (260, 346)  # height, width
    recordings = [
        ("indoor_forward_3_davis_with_gt", 30.0),
        ("indoor_forward_5_davis_with_gt", 30.0),
        ("indoor_forward_6_davis_with_gt", 30.0),
        ("indoor_forward_7_davis_with_gt", 30.0),
        ("indoor_forward_8_davis", 30.0),
        ("indoor_forward_9_davis_with_gt", 30.0),
        ("indoor_forward_10_davis_with_gt", 30.0),
        ("indoor_forward_11_davis", 30.0),
        ("indoor_forward_12_davis", 20.0),
    ]

    # download
    base_url_rec = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/"
    base_url_calib = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv/calib/"

    for rec, t0_skip in recordings:
        name = ("_").join(rec.split("_")[:2])  # eg indoor_forward

        dest = root_dir
        dest.mkdir(parents=True, exist_ok=True)

        # recording
        if not (dest / f"{rec}.h5").exists():
            with tempfile.TemporaryDirectory() as tmp_dir:
                # get files
                tmp_dir = Path(tmp_dir)
                download_and_extract_archive(f"{base_url_calib}{name}_calib_davis.zip", tmp_dir)  # calibration
                download_and_extract_archive(f"{base_url_rec}{rec}.zip", tmp_dir)  # recording

                def append(dataset, data):
                    n = len(data)
                    if n == 0:
                        return
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
                        compression=hdf5plugin.Zstd(),
                    )
                    h5f.create_dataset(
                        "events/t", (0,), maxshape=(None,), chunks=True, dtype=np.float64, compression=hdf5plugin.Zstd()
                    )
                    h5f.create_dataset(
                        "events/y", (0,), maxshape=(None,), chunks=True, dtype=np.uint16, compression=hdf5plugin.Zstd()
                    )
                    h5f.create_dataset(
                        "events/x", (0,), maxshape=(None,), chunks=True, dtype=np.uint16, compression=hdf5plugin.Zstd()
                    )
                    h5f.create_dataset(
                        "events/p", (0,), maxshape=(None,), chunks=True, dtype=np.bool_, compression=hdf5plugin.Zstd()
                    )

                    # if rectifying, also store rectified coordinates
                    if rectify:
                        h5f.create_dataset(
                            "events/y_rect",
                            (0,),
                            maxshape=(None,),
                            chunks=True,
                            dtype=np.float32,
                            compression=hdf5plugin.Zstd(),
                        )
                        h5f.create_dataset(
                            "events/x_rect",
                            (0,),
                            maxshape=(None,),
                            chunks=True,
                            dtype=np.float32,
                            compression=hdf5plugin.Zstd(),
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

                    # write splits to h5 so we can get corresponding raw events
                    frame_splits = np.stack([splits[:-1], splits[1:]], axis=1)
                    h5f.create_dataset(
                        "events/splits", data=frame_splits, chunks=True, dtype=np.int64, compression=hdf5plugin.Zstd()
                    )

                    # precompute backward rectification
                    # kalibr equidistant = .fisheye
                    with open(tmp_dir / f"{name}_calib_davis" / f"camchain-..{name}_calib_davis_cam.yaml", "r") as f:
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

                    # precompute forward rectification
                    w, h = resolution
                    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                    original_coords = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2).astype(np.float32)
                    rect_coords = cv2.fisheye.undistortPoints(original_coords, K_dist, dist_coeffs, P=K_rect)
                    fw_rect_map = rect_coords.reshape(h, w, 2)

                    # store fw/bw rect maps as datasets (too big for attrs)
                    h5f.create_dataset(
                        "fw_rect_map", data=fw_rect_map, chunks=True, dtype=np.float32, compression=hdf5plugin.Zstd()
                    )
                    h5f.create_dataset(
                        "bw_rect_map", data=bw_rect_map, chunks=True, dtype=np.float32, compression=hdf5plugin.Zstd()
                    )

                    # store some useful attributes
                    h5f.attrs["sensor_size"] = sensor_size
                    h5f.attrs["time_window"] = time_window if time_window else False
                    h5f.attrs["count_window"] = count_window if count_window else False
                    h5f.attrs["ts_res"] = ts_res if ts_res else False
                    h5f.attrs["rectify"] = rectify
                    h5f.attrs["K_rect"] = K_rect

                    # convert all to rectified coordinates
                    if rectify:
                        chunk_size = 100000
                        n = len(h5f["events/t"])
                        chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
                        for start, stop in track(chunks, description=f"Rectifying {rec}..."):
                            y = h5f["events/y"][start:stop]
                            x = h5f["events/x"][start:stop]
                            x_rect, y_rect = fw_rect_map[y, x].T
                            append(h5f["events/x_rect"], x_rect)
                            append(h5f["events/y_rect"], y_rect)

                    # convert relevant part to frames
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
                            # channels neg, pos, avg ts (optionally quantized)
                            frame = np.zeros((3, *sensor_size), dtype=np.float32)
                            np.add.at(frame[:2], (p, y, x), 1)
                            np.add.at(frame[-1], (y, x), t_norm)
                            if ts_res:
                                frame[-1] = np.round(frame[-1] / (frame[:2].sum(0) + 1e-9) / ts_res) * ts_res
                            else:
                                frame[-1] = frame[-1] / (frame[:2].sum(0) + 1e-9)

                            # backwards rectification
                            if rectify:
                                frame = cv2.remap(frame.transpose(1, 2, 0), bw_rect_map, None, cv2.INTER_NEAREST)
                                frame = frame.transpose(2, 0, 1)

                            frames.append(frame)

                        # add to h5
                        frames = np.stack(frames)
                        append(h5f["events/frames"], frames)

                    # overwrite raw coords with rectified
                    if rectify:
                        del h5f["events/y"], h5f["events/x"]
                        h5f["events/y"] = h5f["events/y_rect"]
                        h5f["events/x"] = h5f["events/x_rect"]
                        del h5f["events/y_rect"], h5f["events/x_rect"]


if __name__ == "__main__":
    get_uzh_fpv_h5_frames("data/uzh_fpv_10ms_rect", time_window=0.01, count_window=None, ts_res=None, rectify=True)
