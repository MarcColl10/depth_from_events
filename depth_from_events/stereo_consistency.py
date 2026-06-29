import torch
import torch.nn as nn
import torch.nn.functional as F


class StereoEventConsistency(nn.Module):
    """
    Low-overlap stereo event consistency loss.

    This loss uses the predicted left depth to project left-camera pixels into
    the right event image. It only supervises pixels that actually project inside
    the right image, so it is suitable for low-overlap stereo rigs.

    Expected inputs per forward call:
        left_frame:   left event representation,  B x C x H x W
        right_frame:  right event representation, B x C x Hr x Wr
        depth_left:   predicted metric left depth, B x 1 x H x W
        K_left:       left intrinsics,  B x 3 x 3 or 3 x 3
        K_right:      right intrinsics, B x 3 x 3 or 3 x 3
        T_left_to_right:  transform from left camera frame to right camera frame,
                          B x 4 x 4 or 4 x 4
    """

    cls_name = "stereo_event_consistency"

    def __init__(
        self,
        accumulation_window: int,
        weight: float = 0.01,
        patch_size: int = 5,
        min_valid_ratio: float = 0.002,
        eps: float = 1e-6,
        use_census: bool = True,
        T_left_to_right=None,
    ):
        super().__init__()

        self.accumulation_window = accumulation_window
        self.weight = weight
        self.patch_size = patch_size
        self.min_valid_ratio = min_valid_ratio
        self.eps = eps
        self.use_census = use_census

        if T_left_to_right is not None:
            self.register_buffer(
                "T_left_to_right_cfg",
                torch.tensor(T_left_to_right, dtype=torch.float32),
            )
        else:
            self.T_left_to_right_cfg = None

        self.total_loss = 0
        self.passes = 0

        self.left_frames = []
        self.right_frames = []
        self.depths_left = []
        self.K_lefts = []
        self.K_rights = []
        self.T_left_to_rights = []

    def forward(
        self,
        left_frame,
        right_frame,
        depth_left,
        K_left,
        K_right=None,
        T_left_to_right=None,
    ):
        if K_right is None:
            K_right = K_left

        if T_left_to_right is None:
            if self.T_left_to_right_cfg is None:
                raise ValueError(
                    "StereoEventConsistency needs T_left_to_right either "
                    "from the batch or from the loss config."
                )
            T_left_to_right = self.T_left_to_right_cfg

        self.left_frames.append(left_frame)
        self.right_frames.append(right_frame)
        self.depths_left.append(depth_left)
        self.K_lefts.append(K_left)
        self.K_rights.append(K_right)
        self.T_left_to_rights.append(T_left_to_right)

        self.passes += 1

    def backward(self):
        losses = []

        for left_frame, right_frame, depth_left, K_left, K_right, T_lr in zip(
            self.left_frames,
            self.right_frames,
            self.depths_left,
            self.K_lefts,
            self.K_rights,
            self.T_left_to_rights,
        ):
            loss = self._pair_loss(
                left_frame=left_frame,
                right_frame=right_frame,
                depth_left=depth_left,
                K_left=K_left,
                K_right=K_right,
                T_left_to_right=T_lr,
            )

            if loss is not None:
                losses.append(loss)

        if not losses:
            return None

        self.total_loss = torch.stack(losses).mean() * self.weight
        return self.total_loss

    def reset(self):
        self.total_loss = 0
        self.passes = 0

        self.left_frames.clear()
        self.right_frames.clear()
        self.depths_left.clear()
        self.K_lefts.clear()
        self.K_rights.clear()
        self.T_left_to_rights.clear()

    def compute_and_reset(self):
        mean_loss = self.total_loss
        self.reset()
        return {self.cls_name: mean_loss}

    def _pair_loss(
        self,
        left_frame,
        right_frame,
        depth_left,
        K_left,
        K_right,
        T_left_to_right,
    ):
        left_img = self._event_image(left_frame)
        right_img = self._event_image(right_frame)

        grid, valid = self._left_to_right_grid(
            depth_left=depth_left,
            K_left=K_left,
            K_right=K_right,
            T_left_to_right=T_left_to_right,
            right_hw=right_img.shape[-2:],
        )

        right_warped = F.grid_sample(
            right_img,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # Event support mask: ignore completely empty regions.
        support = (left_img.abs() + right_warped.abs()).gt(0)
        valid = valid[:, None] & support

        valid_ratio = valid.float().mean()
        if valid_ratio.detach() < self.min_valid_ratio:
            return None

        if self.use_census:
            left_desc = self._soft_census(left_img)
            right_desc = self._soft_census(right_warped)
        else:
            left_desc = left_img
            right_desc = right_warped

        residual = left_desc - right_desc
        charbonnier = torch.sqrt(residual.pow(2) + self.eps**2)

        valid = valid.expand_as(charbonnier)
        loss = (charbonnier * valid.float()).sum()
        loss = loss / (valid.float().sum() + self.eps)

        return loss

    def _event_image(self, frame):
        """
        Converts the event representation to a single-channel image.

        This assumes the first two channels are positive/negative event-count-like
        channels, which matches the usual event-frame convention in this project.
        """
        if frame.ndim == 3:
            frame = frame.unsqueeze(0)

        if frame.shape[1] >= 2:
            img = frame[:, :2].abs().sum(dim=1, keepdim=True)
        else:
            img = frame.abs().sum(dim=1, keepdim=True)

        # Normalize each image to avoid loss scale depending too much on event rate.
        denom = img.flatten(start_dim=1).mean(dim=1).view(-1, 1, 1, 1)
        img = img / (denom + self.eps)

        return img

    def _soft_census(self, img):
        if self.patch_size <= 1:
            return img

        b, c, h, w = img.shape
        assert c == 1, "Census expects a single-channel image."

        pad = self.patch_size // 2

        patches = F.unfold(
            img,
            kernel_size=self.patch_size,
            padding=pad,
        )

        patches = patches.view(
            b,
            self.patch_size * self.patch_size,
            h,
            w,
        )

        # Differentiable census-like descriptor.
        desc = torch.tanh((patches - img) / 0.1)
        return desc

    def _left_to_right_grid(
        self,
        depth_left,
        K_left,
        K_right,
        T_left_to_right,
        right_hw,
    ):
        b, _, h, w = depth_left.shape
        hr, wr = right_hw
        device = depth_left.device
        dtype = depth_left.dtype

        K_left = self._batch_matrix(K_left, b, device, dtype)
        K_right = self._batch_matrix(K_right, b, device, dtype)
        T_left_to_right = self._batch_matrix(T_left_to_right, b, device, dtype)

        y, x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )

        pix = torch.stack(
            [x.reshape(-1), y.reshape(-1), torch.ones(h * w, device=device, dtype=dtype)],
            dim=0,
        )

        pix = pix.unsqueeze(0).expand(b, -1, -1)

        K_left_inv = torch.inverse(K_left)

        xyz_left = K_left_inv @ pix
        xyz_left = xyz_left * depth_left.reshape(b, 1, -1)

        R = T_left_to_right[:, :3, :3]
        t = T_left_to_right[:, :3, 3:4]

        xyz_right = R @ xyz_left + t

        proj = K_right @ xyz_right
        z = proj[:, 2:3, :].clamp_min(self.eps)

        u = proj[:, 0:1, :] / z
        v = proj[:, 1:2, :] / z

        u = u.reshape(b, h, w)
        v = v.reshape(b, h, w)
        z = z.reshape(b, h, w)

        valid = (
            (z > self.eps)
            & (u >= 0)
            & (u <= wr - 1)
            & (v >= 0)
            & (v <= hr - 1)
        )

        # Normalize for grid_sample.
        u_norm = 2.0 * u / max(wr - 1, 1) - 1.0
        v_norm = 2.0 * v / max(hr - 1, 1) - 1.0

        grid = torch.stack([u_norm, v_norm], dim=-1)

        return grid, valid

    @staticmethod
    def _batch_matrix(matrix, batch_size, device, dtype):
        if not torch.is_tensor(matrix):
            matrix = torch.tensor(matrix, device=device, dtype=dtype)
        else:
            matrix = matrix.to(device=device, dtype=dtype)

        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0).expand(batch_size, -1, -1)

        return matrix
