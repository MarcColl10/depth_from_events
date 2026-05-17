from pathlib import Path
import shutil
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from lightning.pytorch.callbacks import Callback
import numpy as np

from depth_from_events.visualizer import ImageVisualizer, RerunVisualizer


class LiveVisualizer(Callback):
    def __init__(self, app_id, server, web, compression, blueprint=None):
        self.visualizer = RerunVisualizer(app_id, server, web, compression, blueprint)

    def on_batch_end(self, outputs):
        # update blueprint
        all_keys = set()
        [all_keys.update(output.keys()) for output in outputs.values()]
        self.visualizer.update_blueprint(list(all_keys))

        for output in outputs.values():
            self.visualizer.set_counter()

            # things with events
            for k in [k for k in output.keys() if "events" in k]:
                self.visualizer.event_frame(output[k][0].detach().cpu(), name=k)

            # things with flow
            for k in [k for k in output.keys() if "flow" in k and "raw" not in k]:
                self.visualizer.flow_map(output[k][0].detach().cpu(), name=k)

            # things with disparity
            for k in [k for k in output.keys() if "disparity" in k and "raw" not in k]:
                if isinstance(output[k], tuple):
                    self.visualizer.disparity_map(output[k][1][0].detach().cpu(), name=k)
                    self.visualizer.disparity_map(output[k][0][0].detach().cpu(), name=f"gt_{k}")
                else:
                    self.visualizer.disparity_map(output[k][0].detach().cpu(), name=k)

            # things with color
            for k in [k for k in output.keys() if "color" in k]:
                self.visualizer.color_image(output[k][0].detach().cpu(), name=k)

            # things with pose
            for k in [k for k in output.keys() if "pose" in k]:
                self.visualizer.pose_trajectory(output[k][0].detach().cpu(), name=k)

            # for scalar values
            for k in [k for k in output.keys() if isinstance(output[k], (int, float))]:
                self.visualizer.log_scalar(k, output[k])

            # for histograms
            for k in [k for k in output.keys() if k.startswith("hist")]:
                self.visualizer.log_tensor(f"{k}_gt", output[k][0].hist)
                self.visualizer.log_tensor(k, output[k][1].hist)

    def on_train_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)

    def on_validation_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)

    def on_test_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)


class ImageLogger(Callback):
    def __init__(self, root_dir, keys, format):
        self.visualizer = ImageVisualizer(root_dir, keys, format)

    def on_batch_end(self, outputs):
        for output in outputs.values():
            self.visualizer.set_counter()

            # things with events
            for k in [k for k in output.keys() if "events" in k and "raw" not in k]:
                self.visualizer.event_frame(output[k][0].detach().cpu(), name=k)

            # save events as text file
            for k in [k for k in output.keys() if "events_raw" in k]:
                self.visualizer.csv(output[k][0].detach().cpu(), name=k)

            # save as raw numpy array
            for k in [k for k in output.keys() if "raw" in k]:
                self.visualizer.nparray(output[k][0].detach().cpu().numpy(), name=k)

            # things with flow
            for k in [k for k in output.keys() if "flow" in k and "raw" not in k]:
                self.visualizer.flow_map(output[k][0].detach().cpu(), name=k)

            # things with disparity
            for k in [k for k in output.keys() if "disparity" in k and "raw" not in k]:
                if isinstance(output[k], tuple):
                    self.visualizer.disparity_map(output[k][1][0].detach().cpu(), name=k)
                    self.visualizer.disparity_map(output[k][0][0].detach().cpu(), name=f"gt_{k}")
                else:
                    self.visualizer.disparity_map(output[k][0].detach().cpu(), name=k)

            # color images
            for k in [k for k in output.keys() if "color" in k]:
                self.visualizer.color_image(output[k][0].detach().cpu(), name=k)

            # for scalar values
            for k in [k for k in output.keys() if isinstance(output[k], (int, float))]:
                self.visualizer.scalar(k, output[k])

    def on_train_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)

    def on_validation_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)

    def on_test_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)


class StoreDsecEvalDisparity(Callback):
    """
    Write DSEC evaluation results to the file structure given in https://dsec.ifi.uzh.ch/disparity-submission-format/.
    """

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_test_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        for output in outputs.values():

            disparity_keys = [key for key in output.keys() if key.startswith("depth_disparity")]

            for key in disparity_keys:
                rec = batch.recording
                eval_id, eval_disparity = output[key]
                eval_disparity = eval_disparity.cpu().numpy()

                # format following https://dsec.ifi.uzh.ch/disparity-submission-format/
                disp = eval_disparity.astype(np.float64).squeeze((0, 1))  # remove batch and channel dim
                formatted_disp = (disp * 256).astype(np.uint16)

                # write to file
                (self.output_dir / key / rec).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.output_dir / key / rec / f"{eval_id:06d}.png"), formatted_disp)

    def on_test_epoch_end(self, trainer, litmodule):
        shutil.make_archive(self.output_dir, "zip", self.output_dir)

class PosePlotter(Callback):
    """
    Saves predicted pose and ground-truth pose plots after validation.

    Expected pose format:
        [rx, ry, rz, tx, ty, tz]

    where rotation is axis-angle / rotation-vector format.
    """

    def __init__(self, output_dir="pose_plots", max_batches=None, stage="validate"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.max_batches = max_batches
        self.stage = stage

        self.pred_poses = []
        self.gt_poses = []

    def _get(self, obj, key, default=None):
        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj[key] if key in obj else default

        if hasattr(obj, "get"):
            try:
                value = obj.get(key, default)
                return value if value is not None else default
            except Exception:
                pass

        try:
            value = getattr(obj, key)
        except Exception:
            return default

        # DotMap can return another empty DotMap for missing keys.
        # Do not treat that as real data.
        if value.__class__.__name__ == "DotMap" and len(value) == 0:
            return default

        return value
    def _index_time(self, value, i):
        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            return value[i]

        if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] > i:
            return value[i]

        return value

    def _to_numpy(self, value):
        if value is None:
            return None

        # Real PyTorch tensor
        if torch.is_tensor(value):
            return value.detach().cpu().float().numpy()

        # NumPy array
        if isinstance(value, np.ndarray):
            return value.astype(np.float32)

        # Plain Python numeric/list/tuple values
        if isinstance(value, (int, float, list, tuple)):
            try:
                return np.asarray(value, dtype=np.float32)
            except Exception:
                return None

        # Anything else, e.g. empty DotMap from missing GT pose
        return None

    def _to_pose6(self, value, pad_translation_with_nan=True):
        value = self._to_numpy(value)

        if value is None:
            return None

        while value.ndim > 1:
            value = value[0]

        value = value.astype(np.float32)

        if value.shape[-1] >= 6:
            return value[:6]

        if value.shape[-1] == 3:
            if pad_translation_with_nan:
                pad = np.full(3, np.nan, dtype=np.float32)
            else:
                pad = np.zeros(3, dtype=np.float32)

            return np.concatenate([value, pad], axis=0)

        return None

    def _extract_gt_pose(self, batch, i):
        # Full GT pose, if the dataset provides it.
        pose_gt = self._get(batch, "pose")
        pose_gt = self._index_time(pose_gt, i)
        pose_gt = self._to_pose6(pose_gt)

        if pose_gt is not None:
            return pose_gt

        # Rotation-only GT, if available in auxs.
        auxs = self._get(batch, "auxs")
        gt_rotation = self._get(auxs, "gt_rotation")
        gt_rotation = self._index_time(gt_rotation, i)

        return self._to_pose6(gt_rotation, pad_translation_with_nan=True)

    def on_validation_epoch_start(self, trainer, litmodule):
        self.pred_poses = []
        self.gt_poses = []

    def on_validation_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        if self.max_batches is not None and batch_idx >= self.max_batches:
            return

        if outputs is None:
            return

        for i, output in outputs.items():
            # Prefer raw predicted pose if available.
            pred = output.get(f"{self.stage}/pose_pred", None)

            # Fallback to the existing pose key.
            if pred is None:
                pred = output.get(f"{self.stage}/pose", None)

            pred = self._to_pose6(pred, pad_translation_with_nan=False)

            if pred is None:
                continue

            self.pred_poses.append(pred)

            gt = self._extract_gt_pose(batch, i)
            if gt is not None:
                self.gt_poses.append(gt)

    def _rotvec_to_matrix(self, rotvec):
        theta = np.linalg.norm(rotvec)

        if theta < 1e-8:
            return np.eye(3, dtype=np.float32)

        k = rotvec / theta

        K = np.array(
            [
                [0.0, -k[2], k[1]],
                [k[2], 0.0, -k[0]],
                [-k[1], k[0], 0.0],
            ],
            dtype=np.float32,
        )

        return (
            np.eye(3, dtype=np.float32)
            + np.sin(theta) * K
            + (1.0 - np.cos(theta)) * (K @ K)
        )

    def _compose_trajectory(self, poses):
        orientation = np.eye(3, dtype=np.float32)
        position = np.zeros(3, dtype=np.float32)

        trajectory = [position.copy()]

        for pose in poses:
            rotvec = pose[:3]
            translation = pose[3:]

            if np.isnan(translation).any():
                break

            rotation = self._rotvec_to_matrix(rotvec)

            orientation = rotation @ orientation
            position = position + orientation @ translation

            trajectory.append(position.copy())

        return np.stack(trajectory, axis=0)

    def _has_valid_translation(self, poses):
        return poses.shape[1] >= 6 and not np.isnan(poses[:, 3:6]).any()

    def _save_component_plot(self, pred, gt):
        labels = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]

        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
        axes = axes.ravel()

        for j, ax in enumerate(axes):
            ax.plot(pred[:, j], label="pred")

            if gt is not None and not np.isnan(gt[:, j]).all():
                ax.plot(gt[:, j], "--", label="gt")

            ax.set_title(labels[j])
            ax.grid(True)

        axes[0].legend()
        fig.suptitle("Predicted pose vs ground-truth pose")
        fig.tight_layout()

        fig.savefig(self.output_dir / "pose_components.png", dpi=200)
        plt.close(fig)

    def _save_trajectory_plot(self, pred, gt):
        if not self._has_valid_translation(pred):
            return

        pred_traj = self._compose_trajectory(pred)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            pred_traj[:, 0],
            pred_traj[:, 1],
            pred_traj[:, 2],
            label="pred",
        )

        if gt is not None and self._has_valid_translation(gt):
            gt_traj = self._compose_trajectory(gt)

            ax.plot(
                gt_traj[:, 0],
                gt_traj[:, 1],
                gt_traj[:, 2],
                "--",
                label="gt",
            )

        ax.set_title("Composed pose trajectory")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()

        fig.tight_layout()
        fig.savefig(self.output_dir / "pose_trajectory_3d.png", dpi=200)
        plt.close(fig)

    def on_validation_epoch_end(self, trainer, litmodule):
        if len(self.pred_poses) == 0:
            print("PosePlotter: no predicted poses found.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        pred = np.stack(self.pred_poses, axis=0)

        gt = None
        if len(self.gt_poses) == len(self.pred_poses):
            gt = np.stack(self.gt_poses, axis=0)
        elif len(self.gt_poses) > 0:
            print(
                "PosePlotter: GT pose was only found for part of the sequence. "
                "Saving predicted pose only."
            )

        np.save(self.output_dir / "pred_pose.npy", pred)

        if gt is not None:
            np.save(self.output_dir / "gt_pose.npy", gt)

        self._save_component_plot(pred, gt)
        self._save_trajectory_plot(pred, gt)

        print(f"PosePlotter: saved pose plots to {self.output_dir.resolve()}")