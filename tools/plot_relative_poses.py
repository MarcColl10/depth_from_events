from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_pose(path):
    x = np.load(path)
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    if x.shape[-1] != 6:
        raise ValueError(f"{path} has shape {x.shape}, expected Nx6")
    return x.astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pose_dir", type=Path, help="Folder containing pred_pose.npy and gt_pose.npy")
    parser.add_argument("--dt", type=float, default=None, help="Time window in seconds, e.g. 0.01 or 0.02")
    args = parser.parse_args()

    pose_dir = args.pose_dir
    pred_path = pose_dir / "pred_pose.npy"
    gt_path = pose_dir / "gt_pose.npy"

    pred = load_pose(pred_path)
    gt = load_pose(gt_path)

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

    labels = [
        "rx [rad/window]",
        "ry [rad/window]",
        "rz [rad/window]",
        "tx [m/window]",
        "ty [m/window]",
        "tz [m/window]",
    ]

    # 6-component relative pose plot
    fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(x, gt[:, i], label="GT", linestyle="--", linewidth=1.5)
        ax.plot(x, pred[:, i], label="Pred", linewidth=1.2)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel(xlabel)
    fig.suptitle("Relative pose per time window")
    fig.tight_layout()
    fig.savefig(pose_dir / "relative_pose_components.png", dpi=200)
    plt.close(fig)

    # Translation and rotation magnitude per window
    gt_rot_norm = np.linalg.norm(gt[:, 0:3], axis=1)
    pred_rot_norm = np.linalg.norm(pred[:, 0:3], axis=1)

    gt_trans_norm = np.linalg.norm(gt[:, 3:6], axis=1)
    pred_trans_norm = np.linalg.norm(pred[:, 3:6], axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, gt_trans_norm, label="GT translation norm", linestyle="--", linewidth=1.5)
    ax.plot(x, pred_trans_norm, label="Pred translation norm", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("translation magnitude [m/window]")
    ax.set_title("Relative translation magnitude per time window")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pose_dir / "relative_translation_norm.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, np.degrees(gt_rot_norm), label="GT rotation norm", linestyle="--", linewidth=1.5)
    ax.plot(x, np.degrees(pred_rot_norm), label="Pred rotation norm", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rotation magnitude [deg/window]")
    ax.set_title("Relative rotation magnitude per time window")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pose_dir / "relative_rotation_norm.png", dpi=200)
    plt.close(fig)

    # Save numerical relative-pose comparison as CSV
    out = np.column_stack([x, gt, pred, gt_trans_norm, pred_trans_norm, gt_rot_norm, pred_rot_norm])
    header = (
        "time_or_index,"
        "gt_rx,gt_ry,gt_rz,gt_tx,gt_ty,gt_tz,"
        "pred_rx,pred_ry,pred_rz,pred_tx,pred_ty,pred_tz,"
        "gt_translation_norm,pred_translation_norm,"
        "gt_rotation_norm_rad,pred_rotation_norm_rad"
    )
    np.savetxt(pose_dir / "relative_pose_comparison.csv", out, delimiter=",", header=header, comments="")

    print("Saved:")
    print(pose_dir / "relative_pose_components.png")
    print(pose_dir / "relative_translation_norm.png")
    print(pose_dir / "relative_rotation_norm.png")
    print(pose_dir / "relative_pose_comparison.csv")


if __name__ == "__main__":
    main()
