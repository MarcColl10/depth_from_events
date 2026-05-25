import h5py
import cv2
import numpy as np
from pathlib import Path


JOBS = [
    (
        "/data/marc/raw/mvsec/indoor_flying/indoor_flying1_data.hdf5",
        "davis/left/image_raw",
        "mvsec_indoor_flying1_davis_left.mp4",
        30,
    ),
    (
        "/data/marc/raw/mvsec/outdoor_day/outdoor_day1_data.hdf5",
        "davis/left/image_raw",
        "mvsec_outdoor_day1_davis_left.mp4",
        30,
    ),
    (
        "/data/marc/raw/mvsec/outdoor_day/outdoor_day1_data.hdf5",
        "visensor/left/image_raw",
        "mvsec_outdoor_day1_visensor_left.mp4",
        30,
    ),
]


OUT_DIR = Path("/data/marc/dataset_videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_uint8(img):
    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)
    img = np.nan_to_num(img)
    img -= img.min()
    img /= img.max() + 1e-9
    return (255 * img).astype(np.uint8)


for h5_path, key, out_name, fps in JOBS:
    h5_path = Path(h5_path)
    out_path = OUT_DIR / out_name

    print("=" * 80)
    print("Input:", h5_path)
    print("Key:", key)
    print("Output:", out_path)

    with h5py.File(h5_path, "r") as h:
        if key not in h:
            print("Skipping, key not found.")
            continue

        data = h[key]
        n = data.shape[0]
        h_img, w_img = data.shape[1], data.shape[2]

        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w_img, h_img),
            True,
        )

        for i in range(n):
            img = normalize_uint8(data[i])

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = np.squeeze(img)
                img = normalize_uint8(img)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            writer.write(img)

            if (i + 1) % 1000 == 0 or i + 1 == n:
                print(f"  {i + 1}/{n}")

        writer.release()

    print("Saved:", out_path)
