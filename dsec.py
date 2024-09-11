from bisect import bisect_left
from pathlib import Path
import tempfile

import cv2
from dotmap import DotMap
import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from rich.progress import track
from torchvision.datasets.utils import download_and_extract_archive, download_url
import yaml


def get_dsec_h5_frames(root_dir, time_window, count_window, ts_res, rectify):
    """
    Convert DSEC dataset to h5 files containing raw events and event frames.

    Args:
        root_dir (str): Root directory to save the dataset.
        time_window (int): Time window in microseconds to split the dataset into frames.
        count_window (int): Number of events to split the dataset into frames.
        ts_res (float): Timestamp resolution to quantize the average timestamp channel.
        rectify (bool): Whether to rectify the frames using the calibration data.
    """

    # all train recordings excluding night
    # some from training set as validation
    # all test recordings
    recordings = [
        ("interlaken_00_a", "test", False, True),  # name, stage, has flow gt, has disp gt
        ("interlaken_00_b", "test", True, True),
        ("interlaken_00_c", "train", False, True),
        ("interlaken_00_d", "train", False, True),
        ("interlaken_00_e", "train", False, True),
        ("interlaken_00_f", "train", False, True),
        ("interlaken_00_g", "train", False, True),
        ("interlaken_01_a", "test", True, True),
        ("thun_00_a", "val", True, True),  # daylight
        ("thun_01_a", "test", True, True),
        ("thun_01_b", "test", True, True),
        ("zurich_city_01_a", "val", True, True),  # darkish
        ("zurich_city_04_a", "train", False, True),
        ("zurich_city_04_b", "train", False, True),
        ("zurich_city_04_c", "train", False, True),
        ("zurich_city_04_d", "train", False, True),
        ("zurich_city_04_e", "train", False, True),
        ("zurich_city_04_f", "train", False, True),
        ("zurich_city_05_a", "train", True, True),
        ("zurich_city_05_b", "train", True, True),
        ("zurich_city_06_a", "train", True, True),
        ("zurich_city_07_a", "train", True, True),
        ("zurich_city_08_a", "train", True, True),
        ("zurich_city_09_a", "val", True, True),  # night
        ("zurich_city_11_a", "train", True, True),
        ("zurich_city_11_b", "train", True, True),
        ("zurich_city_11_c", "train", True, True),
        ("zurich_city_12_a", "test", True, True),
        ("zurich_city_13_a", "test", False, True),
        ("zurich_city_13_b", "test", False, True),
        ("zurich_city_14_a", "test", False, True),
        ("zurich_city_14_b", "test", False, True),
        ("zurich_city_14_c", "test", True, True),
        ("zurich_city_15_a", "test", True, True),
    ]

    # sensor size
    sensor_size = (480, 640)  # height, width

    # make root directory
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # download files
    base_url = "https://download.ifi.uzh.ch/rpg/DSEC"
    files = DotMap(
        events="events_left.zip",
        calibration="calibration.zip",
        disp_gt="disparity_event.zip",
        disp_ts="disparity_timestamps.txt",
        disp_test="test_disparity_timestamps.zip",
        flow_gt="optical_flow_forward_event.zip",
        flow_ts="optical_flow_forward_timestamps.txt",
        flow_test="test_forward_optical_flow_timestamps.zip",
    )
    for name, stage, has_flow_gt, has_disp_gt in recordings:
        if not (root_dir / f"{name}.h5").exists():
            with tempfile.TemporaryDirectory() as tmp_dir:
                # get files
                tmp_dir = Path(tmp_dir)

                # useful
                def append(dataset, data):
                    n = len(data)
                    if n == 0:
                        return
                    dataset.resize(len(dataset) + n, axis=0)
                    dataset[-n:] = data

                # put everything in same h5
                with h5py.File(root_dir / f"{name}.h5", "w") as h5f:

                    # events and calibration
                    download_and_extract_archive(f"{base_url}/{stage}/{name}/{name}_{files.events}", tmp_dir)
                    download_and_extract_archive(f"{base_url}/{stage}/{name}/{name}_{files.calibration}", tmp_dir)

                    # copy events
                    # TODO: what does this do to compression?
                    with h5py.File(tmp_dir / "events.h5", "r") as h5f_events:
                        h5f_events.copy("events", h5f)
                        t_offset = h5f_events["t_offset"][()]

                    # create other datasets
                    h5f.create_dataset(
                        "events/frames",
                        (0, 3, *sensor_size),
                        maxshape=(None, 3, *sensor_size),
                        chunks=True,
                        dtype=np.float32,
                        compression=hdf5plugin.Zstd(),
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

                    # process into frames
                    if time_window is not None:
                        t0, tk = h5f["events/t"][0], h5f["events/t"][-1]
                        n_full_windows = int((tk - t0) // time_window)
                        t_split = np.linspace(t0, tk, n_full_windows + 1)
                        splits = np.searchsorted(h5f["events/t"], t_split)
                    elif count_window is not None:
                        start = bisect_left(h5f["events/t"], h5f["events/t"][0])
                        splits = np.arange(start, len(h5f["events/t"]), count_window)

                    # write splits to h5 so we can get corresponding raw events
                    frame_splits = np.stack([splits[:-1], splits[1:]], axis=1)
                    h5f.create_dataset(
                        "events/splits", data=frame_splits, chunks=True, dtype=np.int64, compression=hdf5plugin.Zstd()
                    )

                    # copy rectification map
                    with h5py.File(tmp_dir / "rectify_map.h5", "r") as h5f_rect:
                        h5f_rect.copy("rectify_map", h5f, name="fw_rect_map")

                    # compute backward rectifcation map from calibration
                    with open(tmp_dir / "cam_to_cam.yaml") as f:
                        cam_to_cam = yaml.safe_load(f)
                    fx, fy, cx, cy = cam_to_cam["intrinsics"]["cam0"]["camera_matrix"]  # distorted image
                    K_dist = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    fx, fy, cx, cy = cam_to_cam["intrinsics"]["camRect0"]["camera_matrix"]  # rectified image
                    K_rect = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    R_rect = np.array(cam_to_cam["extrinsics"]["R_rect0"])
                    dist_coeffs = np.array(cam_to_cam["intrinsics"]["cam0"]["distortion_coeffs"])
                    resolution = cam_to_cam["intrinsics"]["cam0"]["resolution"]  # xy
                    bw_rect_map, _ = cv2.initUndistortRectifyMap(
                        K_dist, dist_coeffs, R_rect, K_rect, resolution, cv2.CV_32FC2
                    )

                    # put in h5 (too big for attrs)
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
                    h5f.attrs["t_offset"] = t_offset  # TODO: put in events/t? or does this hurt file size?

                    # convert all to rectified coordinates
                    if rectify:
                        chunk_size = 100000
                        n = len(h5f["events/t"])
                        chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
                        fw_rect_map = h5f["fw_rect_map"][()]
                        for start, stop in track(chunks, description=f"Rectifying {name}..."):
                            y = h5f["events/y"][start:stop]
                            x = h5f["events/x"][start:stop]
                            x_rect, y_rect = fw_rect_map[y, x].T
                            append(h5f["events/x_rect"], x_rect)
                            append(h5f["events/y_rect"], y_rect)

                    # convert to frames
                    chunk_size = 100
                    chunks = np.array_split(np.stack([splits[:-1], splits[1:]]), len(splits) // chunk_size, axis=1)
                    for chunk in track(chunks, description=f"Converting {name} to frames..."):
                        starts, stops = chunk
                        frames = []
                        for start, stop in zip(starts, stops):
                            t = h5f["events/t"][start:stop].astype(np.float64)  # uint32 to float64
                            y = h5f["events/y"][start:stop]  # uint16
                            x = h5f["events/x"][start:stop]  # uint16
                            p = h5f["events/p"][start:stop]  # uint8 in {0, 1}

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

                    # # disparity gt
                    # if has_disp_gt and stage != "test":
                    #     download_and_extract_archive(f"{base_url}/{stage}/{name}/{name}_{files.disp_gt}", tmp_dir)
                    #     download_url(f"{base_url}/{stage}/{name}/{name}_{files.disp_ts}", tmp_dir / files.disp_ts)

                    # # disparity test timestamps
                    # if has_disp_gt and stage == "test":
                    #     download_and_extract_archive(f"{base_url}/{files.disp_test}", tmp_dir)

                    # # optical flow gt
                    # if has_flow_gt and stage != "test":
                    #     download_and_extract_archive(f"{base_url}/{stage}/{name}/{name}_{files.flow_gt}", tmp_dir)
                    #     download_url(f"{base_url}/{stage}/{name}/{name}_{files.flow_ts}", tmp_dir / files.flow_ts)

                    # # optical flow test timestamps
                    # if has_flow_gt and stage == "test":
                    #     download_and_extract_archive(f"{base_url}/{files.flow_test}", tmp_dir)


if __name__ == "__main__":
    get_dsec_h5_frames("data/dsec_10ms_rect", time_window=10000, count_window=None, ts_res=None, rectify=True)
