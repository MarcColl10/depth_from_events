from pathlib import Path
import re

import h5py
import hdf5plugin
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


RAW_ROOT = Path("/data/marc/raw/uzh_fpv/indoor_forward")
PROC_ROOT = Path("/data/marc/uzh_fpv_processed")


def read_groundtruth(path: Path):
    # Read header if present
    header = None
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                header = re.split(r"[,\s]+", s.lstrip("#").strip())
                break
            break

    data = np.genfromtxt(path, comments="#", delimiter=None)
    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 8:
        raise ValueError(f"Expected at least 8 columns in {path}, got {data.shape[1]}")

    ts = data[:, 0].astype(np.float64)
    trans = data[:, 1:4].astype(np.float64)

    # Try to infer quaternion order from the header.
    # scipy expects [qx, qy, qz, qw].
    if header is not None:
        names = [h.lower() for h in header]
        q_names = names[4:8]

        if any("qw" in q or q.endswith("_w") for q in q_names[:1]):
            # [qw, qx, qy, qz] -> [qx, qy, qz, qw]
            q = data[:, [5, 6, 7, 4]].astype(np.float64)
        else:
            # assume [qx, qy, qz, qw]
            q = data[:, 4:8].astype(np.float64)
    else:
        # Most common plain format:
        # timestamp tx ty tz qx qy qz qw
        q = data[:, 4:8].astype(np.float64)

    # Normalize quaternions
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(q_norm, 1e-12)

    # Remove duplicate timestamps if any
    unique_ts, unique_idx = np.unique(ts, return_index=True)
    ts = ts[unique_idx]
    trans = trans[unique_idx]
    q = q[unique_idx]

    return ts, trans, q


def maybe_match_time_units(gt_ts, event_ts):
    gt = gt_ts.copy()
    ev_max = float(np.nanmax(event_ts))
    gt_max = float(np.nanmax(gt))

    # If GT is in nanoseconds and events are in seconds
    if gt_max > 1e12 and ev_max < 1e6:
        gt = gt / 1e9

    # If GT is in microseconds and events are in seconds
    elif gt_max > 1e6 and ev_max < 1e6:
        gt = gt / 1e6

    return gt


def make_pose_interpolators(gt_ts, trans, quat_xyzw):
    rots = R.from_quat(quat_xyzw)
    slerp = Slerp(gt_ts, rots)

    def interp_pose(t):
        t = np.clip(t, gt_ts[0], gt_ts[-1])
        p = np.stack([np.interp(t, gt_ts, trans[:, k]) for k in range(3)], axis=-1)
        rot = slerp(t)
        return p, rot

    return interp_pose


def relative_pose_6d(p0, r0, p1, r1):
    # T_rel = inv(T0) @ T1
    R0 = r0.as_matrix()
    R1 = r1.as_matrix()

    R_rel = R0.T @ R1
    t_rel = R0.T @ (p1 - p0)

    rotvec = R.from_matrix(R_rel).as_rotvec()

    # IMPORTANT:
    # datamodule.py expects stored poses as [tx, ty, tz, rx, ry, rz],
    # then converts to sample.pose = [rx, ry, rz, tx, ty, tz].
    return np.concatenate([t_rel, rotvec]).astype(np.float32)


def add_poses_to_file(h5_path: Path):
    recording = h5_path.stem
    raw_dir = RAW_ROOT / recording
    gt_path = raw_dir / "groundtruth.txt"

    if not gt_path.exists():
        print(f"SKIP {h5_path.name}: no {gt_path}")
        return

    print(f"Processing {h5_path.name}")
    print(f"  GT: {gt_path}")

    gt_ts, gt_trans, gt_quat = read_groundtruth(gt_path)

    with h5py.File(h5_path, "r+") as h5:
        splits = h5["events/splits"][:]
        event_t = h5["events/t"]

        gt_ts = maybe_match_time_units(gt_ts, event_t[: min(len(event_t), 1000)])

        interp_pose = make_pose_interpolators(gt_ts, gt_trans, gt_quat)

        poses = []

        for start, stop in splits:
            start = int(start)
            stop = int(stop)

            if stop <= start:
                poses.append(np.zeros(6, dtype=np.float32))
                continue

            t0 = float(event_t[start])
            t1 = float(event_t[stop - 1])
            
            p0, r0 = interp_pose(np.array([t0], dtype=np.float64))
            p1, r1 = interp_pose(np.array([t1], dtype=np.float64))

            pose = relative_pose_6d(p0[0], r0[0], p1[0], r1[0])
            poses.append(pose)

        poses = np.stack(poses).astype(np.float32)

        if "poses" in h5:
            del h5["poses"]

        dset = h5.create_dataset("poses", data=poses)
        dset.attrs["gt_pose_start_idx"] = 0
        dset.attrs["gt_pose_end_idx"] = len(poses)
        dset.attrs["gt_pose_available_frames"] = len(poses)

        print(f"  wrote poses: {poses.shape}")


def main():
    if not PROC_ROOT.exists():
        raise FileNotFoundError(f"Processed root does not exist: {PROC_ROOT}")

    files = sorted(PROC_ROOT.glob("*_with_gt.h5"))

    if not files:
        raise FileNotFoundError(f"No *_with_gt.h5 files found in {PROC_ROOT}")

    for h5_path in files:
        add_poses_to_file(h5_path)


if __name__ == "__main__":
    main()
