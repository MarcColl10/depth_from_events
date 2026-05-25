from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

GT_COLOR = 'tab:orange'
PRED_COLOR = 'tab:blue'
from scipy.spatial.transform import Rotation as R


def load_pose(path):
    x = np.load(path)
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    if x.shape[-1] != 6:
        raise ValueError(f"{path} has shape {x.shape}, expected Nx6")
    return x.astype(float)


def rotvec_to_quat(rotvec):
    # scipy returns quaternion as [qx, qy, qz, qw]
    return R.from_rotvec(rotvec).as_quat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pose_dir", type=Path)
    parser.add_argument("--dt", type=float, default=None)
    args = parser.parse_args()

    pose_dir = args.pose_dir

    pred = load_pose(pose_dir / "pred_pose.npy")
    gt = load_pose(pose_dir / "gt_pose.npy")

    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]

    valid = np.isfinite(gt).all(axis=1) & np.isfinite(pred).all(axis=1)
    pred = pred[valid]
    gt = gt[valid]

    if args.dt is None:
        x = np.arange(len(pred))
        xlabel = "time window index"
    else:
        x = np.arange(len(pred)) * args.dt
        xlabel = "time [s]"

    gt_q = rotvec_to_quat(gt[:, 0:3])
    pred_q = rotvec_to_quat(pred[:, 0:3])

    gt_t = gt[:, 3:6]
    pred_t = pred[:, 3:6]

    # Plot translation per window
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    t_labels = ["tx [m/window]", "ty [m/window]", "tz [m/window]"]

    for i, ax in enumerate(axes):
        ax.plot(x, gt_t[:, i], label="GT", color=GT_COLOR, linestyle="--", linewidth=1.5)
        ax.plot(x, pred_t[:, i], label="Pred", color=PRED_COLOR, linewidth=1.2)
        ax.set_ylabel(t_labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel(xlabel)
    fig.suptitle("Relative translation per time window")
    fig.tight_layout()
    fig.savefig(pose_dir / "relative_translation_components.png", dpi=200)
    plt.close(fig)

    # Plot quaternion per window
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    q_labels = ["qx", "qy", "qz", "qw"]

    for i, ax in enumerate(axes):
        ax.plot(x, gt_q[:, i], label="GT", color=GT_COLOR, linestyle="--", linewidth=1.5)
        ax.plot(x, pred_q[:, i], label="Pred", color=PRED_COLOR, linewidth=1.2)
        ax.set_ylabel(q_labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel(xlabel)
    fig.suptitle("Relative rotation quaternion per time window")
    fig.tight_layout()
    fig.savefig(pose_dir / "relative_quaternion_components.png", dpi=200)
    plt.close(fig)

    # Translation magnitude per window
    gt_trans_norm = np.linalg.norm(gt_t, axis=1)
    pred_trans_norm = np.linalg.norm(pred_t, axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, gt_trans_norm, label="GT translation norm", color=GT_COLOR, linestyle="--", linewidth=1.5)
    ax.plot(x, pred_trans_norm, label="Pred translation norm", color=PRED_COLOR, linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("translation magnitude [m/window]")
    ax.set_title("Relative translation magnitude per time window")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pose_dir / "relative_translation_norm.png", dpi=200)
    plt.close(fig)

    # Save CSV
    out = np.column_stack([
        x,
        gt_t,
        pred_t,
        gt_q,
        pred_q,
        gt_trans_norm,
        pred_trans_norm,
    ])

    header = (
        "time_or_index,"
        "gt_tx,gt_ty,gt_tz,"
        "pred_tx,pred_ty,pred_tz,"
        "gt_qx,gt_qy,gt_qz,gt_qw,"
        "pred_qx,pred_qy,pred_qz,pred_qw,"
        "gt_translation_norm,pred_translation_norm"
    )

    np.savetxt(
        pose_dir / "relative_pose_quaternion_comparison.csv",
        out,
        delimiter=",",
        header=header,
        comments="",
    )

    print("Saved:")
    print(pose_dir / "relative_translation_components.png")
    print(pose_dir / "relative_quaternion_components.png")
    print(pose_dir / "relative_translation_norm.png")
    print(pose_dir / "relative_pose_quaternion_comparison.csv")


if __name__ == "__main__":
    main()
