from kornia.geometry.depth import depth_to_3d
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_from_events.disparity import DisparityToFlow


class ScaleConsistency(nn.Module):
    """
    Geometric scale consistency loss from Bian et al. 2019.
    Code adapted from https://github.com/JiawangBian/sc_depth_pl/blob/master/losses/loss_functions.py.
    TODO: make nice
    TODO: make more stable with 2-way warping and masking by valid?
    TODO: reuse stuff from disparity.py
    """

    def __init__(self, accumulation_window, weight):
        super().__init__()

        self.accumulation_window = accumulation_window
        self.weight = weight
        self.d2f = DisparityToFlow(min_depth=0.1, max_depth=100.0)  # min and max not used

        self.total_loss = 0
        self.passes = 0
        self.depth_maps = []
        self.poses = []
        self.K_rect = None

    def forward(self, depth_map, pose, K_rect):
        self.depth_maps.append(depth_map)
        self.poses.append(pose)
        self.K_rect = K_rect
        self.passes += 1

    def backward(self):
        for depth_0, depth_1, pose in zip(self.depth_maps, self.depth_maps[1:], self.poses):
            # split pose into axis-angle representation and translation
            axis_angle, translation = pose.split([3, 3], dim=-1)

            # transformation and projection matrix
            transformation = self.d2f.get_transformation_matrix(axis_angle, translation)[:, :3, :]
            projection = (self.K_rect @ transformation)[:, :3, :]

            # backproject depth maps to 3d points
            # NOTE: to prevent all the warnings, comment decorator in depth_to_3d function
            # (warnings library to disable the warning wasn't working for me)
            b, _, h, w = depth_0.shape
            world_points = depth_to_3d(depth_0, self.K_rect)  # depth_to_3d_v2 has bug
            world_points = torch.cat([world_points, torch.ones_like(world_points[:, :1])], dim=1)
            cam_points = projection @ world_points.view(b, 4, -1)

            # project 3d points to pixel coordinates
            pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
            pix_coords = pix_coords.view(b, 2, h, w)
            pix_coords = pix_coords.permute(0, 2, 3, 1)
            pix_coords[..., 0] /= w - 1
            pix_coords[..., 1] /= h - 1
            pix_coords = (pix_coords - 0.5) * 2
            computed_depth = cam_points[:, 2, :].unsqueeze(1).view(b, 1, h, w)

            # sample depth (and image for masking)
            projected_img = F.grid_sample(
                torch.ones_like(depth_1), pix_coords, padding_mode="zeros", align_corners=False
            )
            projected_depth = F.grid_sample(depth_1, pix_coords, padding_mode="zeros", align_corners=False)

            # mask by valid pixels
            valid_mask = projected_img.abs().gt(0).float()

            # scale consistency loss
            diff_depth = (computed_depth - projected_depth).abs() / (computed_depth + projected_depth)
            self.total_loss += (diff_depth * valid_mask).mean()

        self.total_loss /= self.passes
        self.total_loss *= self.weight

        return self.total_loss

    def reset(self):
        self.total_loss = 0
        self.passes = 0
        self.depth_maps.clear()
        self.poses.clear()
        self.K_rect = None

    def compute_and_reset(self):
        mean_loss = self.total_loss
        self.reset()
        return {"scale_consistency": mean_loss}
