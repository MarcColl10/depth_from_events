from pathlib import Path

import h5py
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as R, Slerp


DATA_ROOT = Path("/data/marc/evslam_drone_dataset")
OUT_ROOT = Path("/data/marc/evslam_drone_h5")

EVENT_TOPIC = "/dvxplorer_left/events"
TIME_WINDOW = 0.01  # 10 ms, same style as UZH-FPV processed data

# GT file appears to be:
# t px py pz qx qy qz qw vx vy vz
# If rotations look wrong, change this to "wxyz".
QUAT_ORDER = "xyzw"


def stamp_to_sec(stamp):
    if hasattr(stamp, "sec"):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    if hasattr(stamp, "secs"):
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9
    raise TypeError(f"Unknown stamp format: {type(stamp)}")


def read_events_from_bag(bag_path):
    xs, ys, ps, ts = [], [], [], []

    print(f"Reading events from {bag_path}")

    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic == EVENT_TOPIC]

        if not conns:
            topics = sorted({c.topic for c in reader.connections})
            raise RuntimeError(
                f"Topic {EVENT_TOPIC} not found in {bag_path}. "
                f"Available topics: {topics}"
            )

        for conn, _, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, conn.msgtype)

            if len(msg.events) == 0:
                continue

            xs.append(np.fromiter((e.x for e in msg.events), dtype=np.uint16))
            ys.append(np.fromiter((e.y for e in msg.events), dtype=np.uint16))
            ps.append(np.fromiter((1 if e.polarity else 0 for e in msg.events), dtype=np.uint8))
            ts.append(np.fromiter((stamp_to_sec(e.ts) for e in msg.events), dtype=np.float64))

    if not xs:
        raise RuntimeError(f"No events read from {bag_path}")

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    t = np.concatenate(ts)

    order = np.argsort(t)
    x, y, p, t = x[order], y[order], p[order], t[order]

    height = int(y.max()) + 1
    width = int(x.max()) + 1

    print(f"  events: {len(t)}")
    print(f"  resolution inferred: {width} x {height}")
    print(f"  time range: {t[0]:.6f} -> {t[-1]:.6f}")

    return x, y, p, t, height, width


def read_gt(gt_path):
    print(f"Reading GT from {gt_path}")

    data = np.loadtxt(gt_path)

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 8:
        raise RuntimeError(f"Expected at least 8 columns in {gt_path}, got {data.shape[1]}")

    gt_t = data[:, 0].astype(np.float64)
    pos = data[:, 1:4].astype(np.float64)

    q_raw = data[:, 4:8].astype(np.float64)

    if QUAT_ORDER == "xyzw":
        quat_xyzw = q_raw
    elif QUAT_ORDER == "wxyz":
        quat_xyzw = q_raw[:, [1, 2, 3, 0]]
    else:
        raise ValueError(f"Unsupported QUAT_ORDER: {QUAT_ORDER}")

    quat_xyzw = quat_xyzw / np.maximum(np.linalg.norm(quat_xyzw, axis=1, keepdims=True), 1e-12)

    unique_t, idx = np.unique(gt_t, return_index=True)
    gt_t = unique_t
    pos = pos[idx]
    quat_xyzw = quat_xyzw[idx]

    print(f"  gt poses: {len(gt_t)}")
    print(f"  gt time range: {gt_t[0]:.6f} -> {gt_t[-1]:.6f}")

    return gt_t, pos, quat_xyzw


def build_splits(event_t):
    t0 = event_t[0]
    t1 = event_t[-1]

    edges = np.arange(t0, t1 + TIME_WINDOW, TIME_WINDOW, dtype=np.float64)

    starts = np.searchsorted(event_t, edges[:-1], side="left")
    stops = np.searchsorted(event_t, edges[1:], side="left")

    valid = stops > starts
    splits = np.stack([starts[valid], stops[valid]], axis=1).astype(np.int64)

    return splits


def make_identity_maps(height, width):
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    # OpenCV-style map: H x W x 2, channels are x and y.
    rect_map = np.stack([xs, ys], axis=-1).astype(np.float32)

    return rect_map, rect_map.copy()


def make_pose_interpolator(gt_t, pos, quat_xyzw):
    rot = R.from_quat(quat_xyzw)
    slerp = Slerp(gt_t, rot)

    def interp(query_t):
        query_t = np.asarray(query_t, dtype=np.float64)
        p = np.stack(
            [np.interp(query_t, gt_t, pos[:, k]) for k in range(3)],
            axis=-1,
        )
        r = slerp(query_t)
        return p, r

    return interp


def relative_pose(p0, r0, p1, r1):
    # Relative transform inv(T0) @ T1.
    R0 = r0.as_matrix()
    R1 = r1.as_matrix()

    R_rel = R0.T @ R1
    t_rel = R0.T @ (p1 - p0)

    rotvec = R.from_matrix(R_rel).as_rotvec()

    # Stored format expected by datamodule.py:
    # [tx, ty, tz, rx, ry, rz]
    return np.concatenate([t_rel, rotvec]).astype(np.float32)


def compute_poses_for_splits(event_t, splits, gt_t, pos, quat_xyzw):
    interp = make_pose_interpolator(gt_t, pos, quat_xyzw)

    poses = []

    for start, stop in splits:
        t0 = float(event_t[start])
        t1 = float(event_t[stop - 1])

        if t0 < gt_t[0] or t1 > gt_t[-1]:
            poses.append(np.full(6, np.nan, dtype=np.float32))
            continue

        p0, r0 = interp([t0])
        p1, r1 = interp([t1])

        poses.append(relative_pose(p0[0], r0[0], p1[0], r1[0]))

    return np.stack(poses).astype(np.float32)


def write_h5(out_path, x, y, p, t, splits, poses, height, width):
    print(f"Writing {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fw_rect_map, bw_rect_map = make_identity_maps(height, width)

    with h5py.File(out_path, "w") as h:
        events = h.create_group("events")

        events.create_dataset("x", data=x, compression="gzip", compression_opts=4)
        events.create_dataset("y", data=y, compression="gzip", compression_opts=4)
        events.create_dataset("p", data=p, compression="gzip", compression_opts=4)
        events.create_dataset("t", data=t, compression="gzip", compression_opts=4)
        events.create_dataset("splits", data=splits, compression="gzip", compression_opts=4)

        h.create_dataset("poses", data=poses, compression="gzip", compression_opts=4)

        h.create_dataset("fw_rect_map", data=fw_rect_map, compression="gzip", compression_opts=4)
        h.create_dataset("bw_rect_map", data=bw_rect_map, compression="gzip", compression_opts=4)

        h.attrs["height"] = height
        h.attrs["width"] = width
        h.attrs["time_window"] = TIME_WINDOW
        h.attrs["event_topic"] = EVENT_TOPIC
        h.attrs["quat_order"] = QUAT_ORDER

    print("  done")


def convert_sequence(seq_dir):
    seq = seq_dir.name
    bag_path = seq_dir / f"{seq}.bag"
    gt_path = seq_dir / f"{seq}_gt.txt"
    out_path = OUT_ROOT / f"{seq}.h5"

    if not bag_path.exists() or not gt_path.exists():
        print(f"SKIP {seq}: missing bag or gt")
        return

    x, y, p, t, height, width = read_events_from_bag(bag_path)
    gt_t, pos, quat_xyzw = read_gt(gt_path)

    splits = build_splits(t)
    poses = compute_poses_for_splits(t, splits, gt_t, pos, quat_xyzw)

    print(f"  splits: {splits.shape}")
    print(f"  poses:  {poses.shape}")
    print(f"  valid GT windows: {np.isfinite(poses).all(axis=1).sum()} / {len(poses)}")

    write_h5(out_path, x, y, p, t, splits, poses, height, width)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted((DATA_ROOT / "drone").glob("*"))

    for seq_dir in seq_dirs:
        if seq_dir.is_dir():
            convert_sequence(seq_dir)


if __name__ == "__main__":
    main()
