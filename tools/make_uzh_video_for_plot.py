import hdf5plugin
import h5py
import cv2
import numpy as np
from pathlib import Path

h5_path = Path("/data/marc/uzh_fpv_processed/indoor_forward_10_davis_with_gt.h5")
pose_dir = Path("/data/marc/pose_plots_uzh_fpv_gtpose")
out_path = Path("/data/marc/dataset_videos/uzh_fpv/uzh_fpv_indoor_forward_10_event_windows.mp4")

fps = 100
dt = 0.01  # UZH-FPV processed 10 ms windows

out_path.parent.mkdir(parents=True, exist_ok=True)

pred_path = pose_dir / "pred_pose.npy"
n_pose = None
if pred_path.exists():
    n_pose = len(np.load(pred_path))
    print("Pose windows:", n_pose)

def frame_to_bgr(fr):
    fr = np.asarray(fr)

    # Expected shape: [C, H, W]
    if fr.ndim == 3:
        if fr.shape[0] >= 2:
            neg = fr[0].astype(np.float32)
            pos = fr[1].astype(np.float32)

            H, W = neg.shape
            img = np.zeros((H, W, 3), dtype=np.uint8)

            neg = neg / (neg.max() + 1e-9)
            pos = pos / (pos.max() + 1e-9)

            # negative events blue, positive events red
            img[..., 0] = (255 * neg).astype(np.uint8)
            img[..., 2] = (255 * pos).astype(np.uint8)
            return img

        fr = fr[0]

    fr = fr.astype(np.float32)
    fr = np.nan_to_num(fr)
    fr -= fr.min()
    fr /= fr.max() + 1e-9
    fr = (255 * fr).astype(np.uint8)
    return cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)

with h5py.File(h5_path, "r") as h:
    if "events/frames" not in h:
        raise KeyError(f"{h5_path} does not contain events/frames")

    frames = h["events/frames"]
    n = frames.shape[0]

    if n_pose is not None:
        n = min(n, n_pose)

    sample = frame_to_bgr(frames[0])
    H, W = sample.shape[:2]

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
        True,
    )

    for i in range(n):
        img = frame_to_bgr(frames[i])

        writer.write(img)

        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"{i + 1}/{n}")

    writer.release()

print("Saved:", out_path)
