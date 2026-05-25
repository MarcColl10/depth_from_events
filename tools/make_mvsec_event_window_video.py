import h5py
import cv2
import numpy as np
from pathlib import Path

data_path = Path("/data/marc/raw/mvsec/indoor_flying/indoor_flying1_data.hdf5")
pose_dir = Path("/data/marc/pose_plots_uzh100_selfsup_on_mvsec_indoor1")
out_path = Path("/data/marc/dataset_videos/mvsec_indoor_flying1_event_windows_20ms.mp4")

dt = 0.02
fps = 50  # real-time playback for 20 ms windows
top, left, bottom, right = 0, 1, 192, 345
H = bottom - top
W = right - left

out_path.parent.mkdir(parents=True, exist_ok=True)

pred = np.load(pose_dir / "pred_pose.npy")
n_windows = len(pred)

with h5py.File(data_path, "r") as h:
    events = h["davis/left/events"][:]

# MVSEC event columns are usually [x, y, t, p], but detect robustly.
sample = events[:10000]
t_col = int(np.argmax(sample.max(axis=0)))
t = events[:, t_col].astype(np.float64)

cols = [i for i in range(4) if i != t_col]
x_col = max(cols, key=lambda c: events[:10000, c].max())
remaining = [c for c in cols if c != x_col]
y_col = max(remaining, key=lambda c: events[:10000, c].max())
p_col = [c for c in cols if c not in [x_col, y_col]][0]

x = events[:, x_col].astype(np.int64)
y = events[:, y_col].astype(np.int64)
p = events[:, p_col]

order = np.argsort(t)
t = t[order]
x = x[order]
y = y[order]
p = p[order]

t0 = t[0]

writer = cv2.VideoWriter(
    str(out_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H),
    True,
)

for i in range(n_windows):
    a = t0 + i * dt
    b = a + dt

    s = np.searchsorted(t, a, side="left")
    e = np.searchsorted(t, b, side="left")

    xi = x[s:e] - left
    yi = y[s:e] - top
    pi = p[s:e]

    valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    xi = xi[valid]
    yi = yi[valid]
    pi = pi[valid]

    img = np.zeros((H, W, 3), dtype=np.uint8)

    neg = pi <= 0
    pos = pi > 0

    img[yi[neg], xi[neg], 0] = 255
    img[yi[pos], xi[pos], 2] = 255

    writer.write(img)

    if (i + 1) % 500 == 0 or i + 1 == n_windows:
        print(f"{i + 1}/{n_windows}")

writer.release()
print("Saved:", out_path)
