from bisect import bisect_left
from pathlib import Path
import shutil
import tempfile

import cv2
from dotmap import DotMap
import h5py
import hdf5plugin
import numpy as np
from rich.progress import track
from torchvision.datasets.utils import download_and_extract_archive, download_url
import yaml


DSEC_TRAIN_RECORDINGS = [
    "interlaken_00_c",
    "interlaken_00_d",
    "interlaken_00_e",
    "interlaken_00_f",
    "interlaken_00_g",
    "zurich_city_04_a",
    "zurich_city_04_b",
    "zurich_city_04_c",
    "zurich_city_04_d",
    "zurich_city_04_e",
    "zurich_city_04_f",
    "zurich_city_05_a",
    "zurich_city_05_b",
    "zurich_city_06_a",
    "zurich_city_07_a",
    "zurich_city_08_a",
    "zurich_city_11_a",
    "zurich_city_11_b",
    "zurich_city_11_c",
]
DSEC_VAL_RECORDINGS = [
    "thun_00_a",
    "zurich_city_01_a",
    "zurich_city_09_a",
]


def get_dsec_h5_frames(root_dir, download_dir, time_window, count_window, subsample, ts_res, rectify):
    """
    Convert DSEC dataset to h5 files containing raw events and event frames.

    Args:
        root_dir (str): Root directory to save the processed dataset.
        download_dir (str): Directory to download the raw dataset.
        time_window (int): Time window in microseconds to split the dataset into frames.
        count_window (int): Number of events to split the dataset into frames.
        subsample (int): Subsample factor to "turn off" pixels; 2 would reduce resolution by half.
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

    # sensor size (height, width)
    sensor_size = (480, 640) if subsample is None else (480 // subsample, 640 // subsample)

    # make root directory
    root_dir = Path(root_dir)
    download_dir = Path(download_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # download url/files
    base_url = "https://download.ifi.uzh.ch/rpg/DSEC"
    files = DotMap(
        events=("events_left", ".zip"),
        calibration=("calibration", ".zip"),
        disp_gt=("disparity_event", ".zip"),
        disp_ts=("disparity_timestamps", ".txt"),
        disp_test=("test_disparity_timestamps", ".zip"),
        flow_gt=("optical_flow_forward_event", ".zip"),
        flow_ts=("optical_flow_forward_timestamps", ".txt"),
        flow_test=("test_forward_optical_flow_timestamps", ".zip"),
    )

    # go over recordings
    for name, stage, has_flow_gt, has_disp_gt in recordings:
        # download raw data
        raw_dir = download_dir / name
        raw_dir.mkdir(parents=True, exist_ok=True)

        # convert val to train for downloading
        stage = "train" if stage == "val" else stage

        # go over files
        for fname, ext in files.values():
            try:
                if not (raw_dir / fname).exists():
                    if ext == ".zip":
                        if "test" not in fname:
                            download_and_extract_archive(
                                f"{base_url}/{stage}/{name}/{name}_{fname}{ext}", raw_dir / fname
                            )
                            (raw_dir / fname / f"{name}_{fname}{ext}").unlink()
                        else:
                            (raw_dir / fname).mkdir(parents=True, exist_ok=True)
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                tmp_dir = Path(tmp_dir)
                                download_and_extract_archive(f"{base_url}/{fname}{ext}", tmp_dir)
                                (tmp_dir / f"{name}.csv").rename(raw_dir / fname / f"{name}.csv")
                    else:
                        download_url(f"{base_url}/{stage}/{name}/{name}_{fname}{ext}", raw_dir / fname)
            except OSError:
                pass

            # delete empty directories
            if (raw_dir / fname).exists() and not len(list((raw_dir / fname).iterdir())):
                print(f"Deleting empty directory {raw_dir / fname}...")
                shutil.rmtree(raw_dir / fname)

        # process to h5
        if not (root_dir / f"{name}.h5").exists():
            # useful
            def append(dataset, data):
                n = len(data)
                if n == 0:
                    return
                dataset.resize(len(dataset) + n, axis=0)
                dataset[-n:] = data

            # put everything in same h5
            with h5py.File(root_dir / f"{name}.h5", "w") as h5f:
                with h5py.File(raw_dir / "events_left" / "events.h5", "r") as h5f_events:
                    # get events
                    if subsample is None:  # copy
                        h5f_events.copy("events", h5f)
                    else:  # subsample
                        # create datasets
                        h5f.create_dataset(
                            "events/t",
                            (0,),
                            maxshape=(None,),
                            chunks=True,
                            dtype=np.uint32,
                            compression=hdf5plugin.Zstd(),
                        )
                        h5f.create_dataset(
                            "events/y",
                            (0,),
                            maxshape=(None,),
                            chunks=True,
                            dtype=np.uint16,
                            compression=hdf5plugin.Zstd(),
                        )
                        h5f.create_dataset(
                            "events/x",
                            (0,),
                            maxshape=(None,),
                            chunks=True,
                            dtype=np.uint16,
                            compression=hdf5plugin.Zstd(),
                        )
                        h5f.create_dataset(
                            "events/p",
                            (0,),
                            maxshape=(None,),
                            chunks=True,
                            dtype=np.uint8,
                            compression=hdf5plugin.Zstd(),
                        )

                        # loop over events and subsample
                        chunk_size = 100000
                        n = len(h5f_events["events/t"])
                        chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
                        for start, stop in track(chunks, description=f"Subsampling {name}..."):
                            t = h5f_events["events/t"][start:stop]
                            y = h5f_events["events/y"][start:stop]
                            x = h5f_events["events/x"][start:stop]
                            p = h5f_events["events/p"][start:stop]
                            mask = (y % subsample == 0) & (x % subsample == 0)
                            t, y, x, p = t[mask], y[mask] // subsample, x[mask] // subsample, p[mask]
                            append(h5f["events/t"], t)
                            append(h5f["events/y"], y)
                            append(h5f["events/x"], x)
                            append(h5f["events/p"], p)

                    # get offset for matching to targets
                    t_offset = h5f_events["t_offset"][()]

                # create other datasets
                h5f.create_dataset(
                    "events/frames",
                    (0, 4, *sensor_size),
                    maxshape=(None, 4, *sensor_size),
                    chunks=True,
                    dtype=np.uint8,
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
                if time_window is not None and count_window is not None:  # hybrid windows (time with count cap)
                    t0, tk = h5f["events/t"][0], h5f["events/t"][-1]
                    splits = [0]
                    t = t0
                    i0 = 0
                    while t < tk:
                        i1 = bisect_left(h5f["events/t"], t + time_window)
                        if i1 - i0 > count_window:
                            t = h5f["events/t"][i0 + count_window]
                            i0 += count_window
                        else:
                            t += time_window
                            i0 = i1
                        splits.append(i0)
                elif time_window is not None:  # time windows
                    t0, tk = h5f["events/t"][0], h5f["events/t"][-1]
                    t_split = np.arange(t0, tk, time_window)  # arange fine because ints
                    splits = np.searchsorted(h5f["events/t"], t_split)
                elif count_window is not None:  # count windows
                    splits = np.arange(0, len(h5f["events/t"]), count_window)

                # write splits to h5 so we can get corresponding raw events
                frame_splits = np.stack([splits[:-1], splits[1:]], axis=1)
                h5f.create_dataset(
                    "events/splits", data=frame_splits, chunks=True, dtype=np.int64, compression=hdf5plugin.Zstd()
                )

                # copy rectification map
                with h5py.File(raw_dir / "events_left" / "rectify_map.h5", "r") as h5f_rect:
                    h5f_rect.copy("rectify_map", h5f, name="fw_rect_map")

                # compute backward rectifcation map from calibration
                with open(raw_dir / "calibration" / "cam_to_cam.yaml") as f:
                    cam_to_cam = yaml.safe_load(f)
                fx, fy, cx, cy = cam_to_cam["intrinsics"]["cam0"]["camera_matrix"]  # distorted image
                resolution = cam_to_cam["intrinsics"]["cam0"]["resolution"]  # xy
                if subsample is not None:
                    resolution = [r // subsample for r in resolution]
                    fx, fy, cx, cy = [v / subsample for v in [fx, fy, cx, cy]]
                K_dist = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                fx, fy, cx, cy = cam_to_cam["intrinsics"]["camRect0"]["camera_matrix"]  # rectified image
                if subsample is not None:
                    fx, fy, cx, cy = [v / subsample for v in [fx, fy, cx, cy]]
                K_rect = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                R_rect = np.array(cam_to_cam["extrinsics"]["R_rect0"])
                dist_coeffs = np.array(cam_to_cam["intrinsics"]["cam0"]["distortion_coeffs"])
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
                h5f.attrs["subsample"] = subsample if subsample else False
                h5f.attrs["ts_res"] = ts_res
                h5f.attrs["rectify"] = rectify
                h5f.attrs["K_rect"] = K_rect
                h5f.attrs["t_offset"] = t_offset  # for matching with targets

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
                            frame = np.zeros((4, *sensor_size), dtype=np.float32)
                            frames.append(frame)
                            continue

                        # normalize timestamp
                        t_norm = (t - t[0]) / (t[-1] - t[0])

                        # make into event count frame
                        # channels neg, pos, avg quantized ts per pol
                        # multiply by ts_res when loading so we can store in uint8
                        frame = np.zeros((4, *sensor_size), dtype=np.float32)
                        np.add.at(frame[:2], (p, y, x), 1)
                        np.add.at(frame[2:], (p, y, x), t_norm)
                        frame[2:] = np.round(frame[2:] / (frame[:2] + 1e-9) / ts_res)

                        # backwards rectification
                        if rectify:
                            frame = cv2.remap(frame.transpose(1, 2, 0), bw_rect_map, None, cv2.INTER_NEAREST)
                            frame = frame.transpose(2, 0, 1)

                        frames.append(frame)

                    # convert to uint16 (uint8 not enough) and add to h5
                    frames = np.stack(frames)
                    append(h5f["events/frames"], frames.astype(np.uint16))

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
    time_window = 10000
    count_window = 100000
    time_window_str = f"{int(time_window / 1000)}ms" if time_window is not None else ""
    count_window_str = f"{count_window}c" if count_window is not None else ""
    subsample = None
    ts_res = 0.25
    rectify = True
    root_dir = f"data/dsec_{time_window_str}{count_window_str}{'_subs' + str(subsample) if subsample is not None else ''}_{ts_res}ts{'_rect' if rectify else ''}"
    get_dsec_h5_frames(
        root_dir,
        download_dir="data/raw/dsec",
        time_window=time_window,
        count_window=count_window,
        subsample=subsample,
        ts_res=ts_res,
        rectify=rectify,
    )
