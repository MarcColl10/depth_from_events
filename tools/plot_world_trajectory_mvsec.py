import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp
from pathlib import Path

pose_dir = Path("/data/marc/pose_plots_uzh100_selfsup_on_mvsec_indoor1")
data_h5 = "/data/marc/raw/mvsec/indoor_flying/indoor_flying1_data.hdf5"
gt_h5 = "/data/marc/raw/mvsec/indoor_flying/indoor_flying1_gt.hdf5"

out_path = pose_dir / "world_trajectory_pred_vs_gt.png"

dt = 0.02
scale_translation = 1.0  # change to 3.47 if you want scale-aligned prediction

pred_rel = np.load(pose_dir / "pred_pose.npy")
gt_rel_saved = np.load(pose_dir / "gt_pose.npy")

n = min(len(pred_rel), len(gt_rel_saved))
pred_rel = pred_rel[:n]

with h5py.File(data_h5, "r") as f:
    events = f["davis/left/events"]
    sample = events[:10000]
    t_col = int(np.argmax(sample.max(axis=0)))
    event_t0 = float(events[0, t_col])

with h5py.File(gt_h5, "r") as f:
    pose_ts = f["davis/left/pose_ts"][:]
    poses = f["davis/left/pose"][:]

gt_pos_abs = poses[:, :3, 3]
gt_rot_abs = R.from_matrix(poses[:, :3, :3])
slerp = Slerp(pose_ts, gt_rot_abs)

query_t = event_t0 + np.arange(n) * dt
query_t = np.clip(query_t, pose_ts[0], pose_ts[-1])

# Absolute GT in world coordinates, sampled at the same event-window timestamps.
gt_world = np.stack(
    [np.interp(query_t, pose_ts, gt_pos_abs[:, k]) for k in range(3)],
    axis=1,
)

# Start prediction from the first GT absolute pose.
p0 = gt_world[0]
R0 = slerp([query_t[0]])[0].as_matrix()

T_pred = np.eye(4)
T_pred[:3, :3] = R0
T_pred[:3, 3] = p0

pred_world = [T_pred[:3, 3].copy()]

for i in range(n - 1):
    rx, ry, rz, tx, ty, tz = pred_rel[i]

    T_rel = np.eye(4)
    T_rel[:3, :3] = R.from_rotvec([rx, ry, rz]).as_matrix()
    T_rel[:3, 3] = scale_translation * np.array([tx, ty, tz])

    # Compose local relative motion into world coordinates.
    T_pred = T_pred @ T_rel
    pred_world.append(T_pred[:3, 3].copy())

pred_world = np.asarray(pred_world)

# Put both trajectories at the same origin for easier comparison.
gt_plot = gt_world - gt_world[0]
pred_plot = pred_world - pred_world[0]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot(pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2], label="pred", linewidth=2)
ax.plot(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], label="gt", linestyle="--", linewidth=2)

ax.scatter(gt_plot[0, 0], gt_plot[0, 1], gt_plot[0, 2], marker="o", label="start")
ax.scatter(gt_plot[-1, 0], gt_plot[-1, 1], gt_plot[-1, 2], marker="x", label="gt end")
ax.scatter(pred_plot[-1, 0], pred_plot[-1, 1], pred_plot[-1, 2], marker="x", label="pred end")

ax.set_xlabel("world x [m]")
ax.set_ylabel("world y [m]")
ax.set_zlabel("world z [m]")
ax.set_title(f"World-frame trajectory, translation scale = {scale_translation}")
ax.legend()

# Equal-ish axis scaling
all_pts = np.vstack([gt_plot, pred_plot])
mins = all_pts.min(axis=0)
maxs = all_pts.max(axis=0)
centers = 0.5 * (mins + maxs)
radius = 0.5 * np.max(maxs - mins)

ax.set_xlim(centers[0] - radius, centers[0] + radius)
ax.set_ylim(centers[1] - radius, centers[1] + radius)
ax.set_zlim(centers[2] - radius, centers[2] + radius)

fig.tight_layout()
fig.savefig(out_path, dpi=250)
plt.close(fig)

print("Saved:", out_path)
print("GT final displacement:", gt_plot[-1])
print("Pred final displacement:", pred_plot[-1])
