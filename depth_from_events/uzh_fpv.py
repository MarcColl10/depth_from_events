from bisect import bisect_left
from pathlib import Path

import cv2
import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from rich.progress import track
from scipy.spatial.transform import Rotation as R
from torchvision.datasets.utils import download_and_extract_archive
import yaml


UZH_FPV_TRAIN_RECORDINGS = [
    "indoor_forward_3_davis_with_gt",
    "indoor_forward_5_davis_with_gt",
    "indoor_forward_6_davis_with_gt",
    "indoor_forward_7_davis_with_gt",
    "indoor_forward_8_davis",
    "indoor_forward_9_davis_with_gt",
    # "indoor_forward_10_davis_with_gt",
    "indoor_forward_11_davis",
    "indoor_forward_12_davis",
]
UZH_FPV_VAL_RECORDINGS = [
    "indoor_forward_10_davis_with_gt",
]


def get_uzh_fpv_h5_frames(root_dir, download_dir, time_window, count_window, crop, subsample, ts_res, rectify):
    """
    Convert UZH-FPV dataset to h5 files containing raw events and event frames.

    Args:
        root_dir (str): Root directory to save the processed dataset.
        download_dir (str): Directory to download the raw dataset.
        time_window (float): Time window in seconds to split the dataset into frames.
        count_window (int): Number of events to split the dataset into frames.
        crop (tuple): Crop the sensor to (top, left, bottom, right).
        subsample (int): Subsample factor to "turn off" pixels; 2 would reduce resolution by half.
        ts_res (float): Timestamp resolution to quantize the average timestamp channel.
        rectify (bool): Whether to rectify the frames using the calibration data.
    """

    # now only indoor forward, but there's also 45deg and outdoor
    # time in seconds to skip at the beginning to approx. start at takeoff
    root_dir = Path(root_dir)
    download_dir = Path(download_dir)
    sensor_size = (260, 346) if crop is None else (crop[2] - crop[0], crop[3] - crop[1])  # height, width
    if subsample is not None:
        sensor_size = (sensor_size[0] // subsample, sensor_size[1] // subsample)
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

    # make root directory
    root_dir.mkdir(parents=True, exist_ok=True)

    # download urls
    base_url_rec = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/"
    base_url_calib = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv/calib/"

    # go over recordings
    for rec, t0_skip in recordings:
        name = ("_").join(rec.split("_")[:2])  # eg indoor_forward

        # download raw data
        raw_dir = download_dir / name
        raw_dir.mkdir(parents=True, exist_ok=True)
        if not (raw_dir / rec).exists():  # recording
            download_and_extract_archive(f"{base_url_rec}{rec}.zip", raw_dir / rec)
            (raw_dir / rec / f"{rec}.zip").unlink()
        if not (raw_dir / "calib").exists():  # calibration
            download_and_extract_archive(f"{base_url_calib}{name}_calib_davis.zip", raw_dir / "calib")
            (raw_dir / "calib" / f"{name}_calib_davis.zip").unlink()

        # process to h5
        if not (root_dir / f"{rec}.h5").exists():
            # handy
            def append(dataset, data):
                n = len(data)
                if n == 0:
                    return
                dataset.resize(len(dataset) + n, axis=0)
                dataset[-n:] = data

            # first get raw data so we can work with it efficiently
            # then make frames with channels: neg count, pos count, avg quantized ts (neg, pos)
            # frames in uint8 to save space
            with h5py.File(root_dir / f"{rec}.h5", "w") as h5f:
                h5f.create_dataset(
                    "events/frames",
                    (0, 4, *sensor_size),
                    maxshape=(None, 4, *sensor_size),
                    chunks=True,
                    dtype=np.uint8,
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
                    "events/p", (0,), maxshape=(None,), chunks=True, dtype=np.uint8, compression=hdf5plugin.Zstd()
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
                    raw_dir / rec / "events.txt",
                    delimiter=" ",
                    skiprows=1,
                    names=["t", "x", "y", "p"],
                    chunksize=1e6,
                )

                # put in raw
                for df in track(events, description=f"Converting {rec} to h5..."):
                    # crop: correct xy value
                    if crop is not None:
                        df = df.loc[
                            (df["x"] >= crop[1]) & (df["x"] < crop[3]) & (df["y"] >= crop[0]) & (df["y"] < crop[2])
                        ]
                        df.loc[:, "x"] -= crop[1]
                        df.loc[:, "y"] -= crop[0]
                    # subsample: filter by xy value
                    if subsample is not None:
                        df = df.loc[(df["x"] % subsample == 0) & (df["y"] % subsample == 0)]
                        df.loc[:, "x"] = df["x"] // subsample
                        df.loc[:, "y"] = df["y"] // subsample

                    # append to h5
                    append(h5f["events/t"], df["t"].values)
                    append(h5f["events/y"], df["y"].values)
                    append(h5f["events/x"], df["x"].values)
                    append(h5f["events/p"], df["p"].values)

                # proces into frames
                if time_window is not None:
                    t0, tk = h5f["events/t"][0], h5f["events/t"][-1]
                    t0 += t0_skip
                    n_full_windows = int((tk - t0) // time_window)
                    t_split = np.linspace(t0, n_full_windows * time_window + t0, n_full_windows + 1)
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
                with open(
                    raw_dir / "calib" / f"{name}_calib_davis" / f"camchain-..{name}_calib_davis_cam.yaml", "r"
                ) as f:
                    cam_to_cam = yaml.safe_load(f)
                fx, fy, cx, cy = cam_to_cam["cam0"]["intrinsics"]
                resolution = cam_to_cam["cam0"]["resolution"]  # xy
                if crop is not None:
                    resolution = [crop[3] - crop[1], crop[2] - crop[0]]
                    cx -= crop[1]
                    cy -= crop[0]
                if subsample is not None:
                    resolution = [r // subsample for r in resolution]
                    fx, fy, cx, cy = [v / subsample for v in [fx, fy, cx, cy]]
                K_dist = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                K_rect = K_dist.copy()  # usually same for fisheye
                dist_coeffs = np.array(cam_to_cam["cam0"]["distortion_coeffs"])  # not affected by subsample
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
                h5f.attrs["subsample"] = subsample if subsample else False
                h5f.attrs["ts_res"] = ts_res
                h5f.attrs["rectify"] = rectify
                h5f.attrs["K_rect"] = K_rect
                h5f.attrs["t_offset"] = 0

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

                    # convert to uint8 and add to h5
                    frames = np.stack(frames)
                    assert frames.min() >= 0 and frames.max() <= 255
                    append(h5f["events/frames"], frames.astype(np.uint8))

                # overwrite raw coords with rectified
                if rectify:
                    del h5f["events/y"], h5f["events/x"]
                    h5f["events/y"] = h5f["events/y_rect"]
                    h5f["events/x"] = h5f["events/x_rect"]
                    del h5f["events/y_rect"], h5f["events/x_rect"]

                # write pose gt
                if (raw_dir / rec / "groundtruth_corrected.txt").exists():
                    pose = pd.read_csv(
                        raw_dir / rec / "groundtruth_corrected.txt",
                        delimiter=" ",
                        skiprows=1,
                        names=["t", "tx", "ty", "tz", "qx", "qy", "qz", "qw"],
                    )

                    # interpolate pose to frame timestamps
                    query_timestamps = h5f["events/t"][:][splits]
                    # limit timestamps to timespan where pose is available
                    have_gt_pose = (pose["t"].iloc[0] <= query_timestamps) & (query_timestamps <= pose["t"].iloc[-1])
                    gt_pose_start_idx = np.argmax(have_gt_pose)
                    gt_pose_available_frames = np.sum(have_gt_pose)
                    gt_pose_end_idx = gt_pose_start_idx + gt_pose_available_frames

                    interpolated_pose = dict()
                    for key, value in pose.items():
                        if key == "t":
                            continue
                        interpolated_pose[key] = np.interp(query_timestamps[have_gt_pose], pose["t"], value)
                    interpolated_pose["t"] = query_timestamps[have_gt_pose]
                    interpolated_pose = pd.DataFrame(interpolated_pose)

                    # compute delta poses
                    poses = []
                    for (_, pose_start), (_, pose_end) in track(
                        zip(interpolated_pose[:-1].iterrows(), interpolated_pose[1:].iterrows()),
                        description=f"Computing delta poses for {rec}...",
                    ):
                        # Compute delta translation
                        delta_translation = pose_end[["tx", "ty", "tz"]].values - pose_start[["tx", "ty", "tz"]].values

                        # Compute delta rotation
                        q1 = pose_start[["qx", "qy", "qz", "qw"]].values
                        q2 = pose_end[["qx", "qy", "qz", "qw"]].values
                        r1 = R.from_quat(q1)
                        r2 = R.from_quat(q2)
                        delta_rotation = r1.inv() * r2
                        delta_rotation_axis_angle = delta_rotation.as_rotvec()

                        # delta translation in body frame
                        delta_translation_body_frame = r1.apply(delta_translation, inverse=True)

                        # assemble to 6D pose
                        poses.append(np.concatenate([delta_translation_body_frame, delta_rotation_axis_angle]))

                    poses = np.stack(poses)
                    poses_h5 = h5f.create_dataset("poses", data=poses, chunks=True, dtype=np.float32, compression=None)
                    poses_h5.attrs["gt_pose_start_idx"] = gt_pose_start_idx
                    poses_h5.attrs["gt_pose_end_idx"] = gt_pose_end_idx
                    poses_h5.attrs["gt_pose_available_frames"] = gt_pose_available_frames


if __name__ == "__main__":
    time_window = 0.01
    crop = [2, 13, 258, 333]
    subsample = 2
    ts_res = 0.25
    rectify = True
    root_dir = f"data/uzh_fpv_{int(time_window * 1000)}ms{'_crop' + str(crop) if crop is not None else ''}{'_subs' + str(subsample) if subsample is not None else ''}_{ts_res}ts{'_rect' if rectify else ''}"
    print(f"Processing UZH-FPV dataset to {root_dir}...")
    get_uzh_fpv_h5_frames(
        root_dir,
        download_dir="data/raw/uzh_fpv",
        time_window=time_window,
        count_window=None,
        crop=crop,
        subsample=subsample,
        ts_res=ts_res,
        rectify=rectify,
    )
