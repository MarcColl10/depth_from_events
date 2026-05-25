from pathlib import Path

import h5py
import numpy as np


ROOT = Path("/data/marc/evslam_drone_h5")

# Real EvSLAM left DVXplorer calibration from:
# /data/marc/evslam_drone_dataset/calibration/calib_others/calib_results_cam_drone.yaml
WIDTH = 640
HEIGHT = 480

K_RECT = np.array(
    [
        [431.98375640237305, 0.0, 319.64858384305677],
        [0.0, 431.61793148581944, 242.9093226796089],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

DISTORTION = np.array(
    [
        0.03363443711669938,
        -0.046237102612126595,
        -0.00013760389998625398,
        0.0001582064761022624,
    ],
    dtype=np.float32,
)


def create_frames(h5):
    events = h5["events"]

    x_all = events["x"]
    y_all = events["y"]
    p_all = events["p"]
    splits = events["splits"][:]

    height, width = map(int, h5.attrs["sensor_size"])

    if "frames" in events:
        del events["frames"]

    frames = events.create_dataset(
        "frames",
        shape=(len(splits), 2, height, width),
        dtype=np.uint8,
        chunks=(1, 2, height, width),
        compression="gzip",
        compression_opts=4,
    )

    for i, (start, stop) in enumerate(splits):
        start = int(start)
        stop = int(stop)

        x = x_all[start:stop].astype(np.int64)
        y = y_all[start:stop].astype(np.int64)
        p = p_all[start:stop].astype(np.int64)

        frame = np.zeros((2, height, width), dtype=np.uint16)

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[valid]
        y = y[valid]
        p = p[valid]

        neg = p == 0
        pos = p != 0

        np.add.at(frame[0], (y[neg], x[neg]), 1)
        np.add.at(frame[1], (y[pos], x[pos]), 1)

        frames[i] = np.clip(frame, 0, 255).astype(np.uint8)

        if (i + 1) % 250 == 0 or i + 1 == len(splits):
            print(f"    frames {i + 1}/{len(splits)}")


def patch_file(path):
    print("=" * 80)
    print(path)

    with h5py.File(path, "r+") as h5:
        h5.attrs["sensor_size"] = np.array([HEIGHT, WIDTH], dtype=np.int64)
        h5.attrs["K_rect"] = K_RECT
        h5.attrs["D"] = DISTORTION
        h5.attrs["distortion_model"] = "radtan"
        h5.attrs["camera_model"] = "pinhole"
        h5.attrs["ts_res"] = np.float32(1.0)

        print("  sensor_size:", h5.attrs["sensor_size"])
        print("  K_rect:")
        print(h5.attrs["K_rect"])
        print("  D:", h5.attrs["D"])

        create_frames(h5)

        print("  events keys:", list(h5["events"].keys()))

        if "poses" in h5:
            print("  poses:", h5["poses"].shape)
        else:
            print("  WARNING: no poses dataset")


def main():
    files = sorted(ROOT.glob("*.h5"))

    if not files:
        raise FileNotFoundError(f"No .h5 files found in {ROOT}")

    for path in files:
        patch_file(path)


if __name__ == "__main__":
    main()
