import torch
import torch.nn as nn


class ForwardBackwardPoseConsistency(nn.Module):
    """
    Forward-backward pose consistency loss.

    For a temporal window, we already have forward poses predicted by the normal
    training loop. This loss runs the same network over that window in reverse
    order and encourages:

        T_forward * T_backward ≈ I

    This is inspired by forward-backward pose consistency constraints used in
    self-supervised monocular depth + ego-motion learning.
    """

    def __init__(
        self,
        accumulation_window,
        weight=0.001,
        rotation_weight=1.0,
        translation_weight=0.1,
    ):
        super().__init__()
        self.accumulation_window = accumulation_window
        self.weight = weight
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

        self.frames = []
        self.forward_poses = []
        self.total_loss = 0
        self.passes = 0

    def forward(self, frame, pose):
        if pose is None:
            return

        self.frames.append(frame)
        self.forward_poses.append(pose)
        self.passes += 1

    def backward(self, network):
        if len(self.forward_poses) == 0:
            return None

        # Save current recurrent state from the normal forward pass.
        saved_state = getattr(network, "state", None)

        # Run the same network backwards over the current accumulation window.
        network.reset()
        backward_poses_reversed = []

        for frame in reversed(self.frames):
            frame_rev = self.reverse_event_frame(frame)
            yhat = network(frame_rev)

            pose_bwd = self.extract_pose(yhat)
            backward_poses_reversed.append(pose_bwd)

        # Restore original recurrent state so the main training loop stays intact.
        network.reset()
        network.state = saved_state

        backward_poses = list(reversed(backward_poses_reversed))

        losses = []

        for pose_fwd, pose_bwd in zip(self.forward_poses, backward_poses):
            T_fwd = self.pose_to_matrix(pose_fwd)
            T_bwd = self.pose_to_matrix(pose_bwd)

            T_cycle = T_fwd @ T_bwd

            eye = torch.eye(
                4,
                device=T_cycle.device,
                dtype=T_cycle.dtype,
            ).unsqueeze(0)

            rot_loss = (T_cycle[:, :3, :3] - eye[:, :3, :3]).abs().mean()
            trans_loss = T_cycle[:, :3, 3].abs().mean()

            loss = (
                self.rotation_weight * rot_loss
                + self.translation_weight * trans_loss
            )

            losses.append(loss)

        loss = torch.stack(losses).mean()
        self.total_loss = self.weight * loss

        return self.total_loss

    @staticmethod
    def extract_pose(yhat):
        if isinstance(yhat, (tuple, list)):
            if len(yhat) == 2:
                _, pose = yhat
            elif len(yhat) == 3:
                _, pose, _ = yhat
            else:
                raise ValueError(f"Unexpected network output length: {len(yhat)}")
            return pose

        raise ValueError("ForwardBackwardPoseConsistency requires a depth-pose network.")

    @staticmethod
    def reverse_event_frame(frame):
        """
        Reverse the internal timestamp channels if the input has them.

        In the default repo config, return_events=True, so frames usually have
        only 2 channels: negative and positive event counts. In that case,
        reversing the sequence order is enough.

        If return_events=False, the frame has timestamp channels after the first
        two polarity channels, so we invert them with 1 - t.
        """
        if frame.shape[1] <= 2:
            return frame

        frame_rev = frame.clone()
        frame_rev[:, 2:] = 1.0 - frame_rev[:, 2:]
        return frame_rev

    @staticmethod
    def pose_to_matrix(pose):
        axis_angle, translation = pose.split([3, 3], dim=-1)
        rotation = ForwardBackwardPoseConsistency.rodrigues(axis_angle)

        b = pose.shape[0]
        T = torch.eye(
            4,
            device=pose.device,
            dtype=pose.dtype,
        ).unsqueeze(0).repeat(b, 1, 1)

        T[:, :3, :3] = rotation
        T[:, :3, 3] = translation

        return T

    @staticmethod
    def rodrigues(axis_angle):
        angle = axis_angle.norm(2, dim=-1, keepdim=True)
        axis = axis_angle / (angle + 1e-6)

        b = axis.shape[0]
        axis_cross = torch.zeros(
            b,
            3,
            3,
            device=axis.device,
            dtype=axis.dtype,
        )

        axis_cross[:, 0, 1] = -axis[:, 2]
        axis_cross[:, 0, 2] = axis[:, 1]
        axis_cross[:, 1, 0] = axis[:, 2]
        axis_cross[:, 1, 2] = -axis[:, 0]
        axis_cross[:, 2, 0] = -axis[:, 1]
        axis_cross[:, 2, 1] = axis[:, 0]

        eye = torch.eye(
            3,
            device=axis.device,
            dtype=axis.dtype,
        ).unsqueeze(0)

        angle = angle.unsqueeze(-1)

        rotation = (
            eye
            + torch.sin(angle) * axis_cross
            + (1.0 - torch.cos(angle)) * (axis_cross @ axis_cross)
        )

        return rotation

    def reset(self):
        self.frames.clear()
        self.forward_poses.clear()
        self.total_loss = 0
        self.passes = 0

    def compute_and_reset(self):
        mean_loss = self.total_loss
        self.reset()
        return {"forward_backward_pose": mean_loss}
