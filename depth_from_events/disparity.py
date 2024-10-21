"""
Based on https://github.com/nianticlabs/monodepth2.
"""

import torch
import torch.nn as nn


class DisparityToFlow(nn.Module):
    """
    Convert disparity and pose to flow, optionally with learned intrinsics.

    Intrinsic matrix representation based on
    https://github.com/google-research/google-research/blob/master/depth_and_motion_learning/intrinsics_utils.py.
    """

    def __init__(self, min_depth, max_depth):
        super().__init__()

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.height, self.width = 0, 0

    def init_grid(self, b, h, w, device, dtype):
        self.height, self.width = h, w

        # grid for pixel displacement computation
        x = torch.arange(self.width, device=device, dtype=dtype)
        y = torch.arange(self.height, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.grid = torch.stack([grid_x, grid_y]).unsqueeze(0).expand(b, -1, -1, -1)

        # pixel coordinates for projection
        self.ones = torch.ones(b, 1, self.height * self.width, device=device, dtype=dtype)
        pix_coords = self.grid.view(b, 2, -1)
        self.pix_coords = torch.cat([pix_coords, self.ones], dim=1)

    def forward(self, prediction, K_rect, inv_K_rect):
        # unpack
        if len(prediction) == 2:
            disparity, pose = prediction
        elif len(prediction) == 3:
            disparity, pose, intrinsics = prediction
            fx_factor, fy_factor, cx_factor, cy_factor = intrinsics.unbind(-1)
            b, _, h, w = disparity.shape
            K_rect = torch.zeros(b, 3, 3, device=disparity.device, dtype=disparity.dtype)
            K_rect[:, 0, 0] = fx_factor * (h + w)  # this and fy had 0.5 in it, but sigmoid now
            K_rect[:, 1, 1] = fy_factor * (h + w)
            K_rect[:, 0, 2] = cx_factor * w
            K_rect[:, 1, 2] = cy_factor * h
            K_rect[:, 2, 2] = 1
            inv_K_rect = torch.linalg.pinv(K_rect)

        # split pose into axis-angle representation and translation
        axis_angle, translation = pose.split([3, 3], dim=-1)

        # transformation matrix (oldest -> most recent event ts)
        transformation = self.get_transformation_matrix(axis_angle, translation)

        # convert disparity to depth
        # _, depth = self.disparity_to_depth(disparity)
        depth = disparity

        # backproject depth maps to 3d points
        cam_points = self.backproject_depth(depth, inv_K_rect)

        # project 3d points to new cam location
        new_pix_coords = self.project_3d(cam_points, transformation, K_rect)

        # compute flow, convert -xy to xy
        flow = -(new_pix_coords - self.grid)

        return flow

    def project_3d(self, cam_points, transformation, K_rect):
        """
        Project 3d points to new camera location.
        """
        # transform 3d points
        new_cam_points = transformation @ cam_points

        # project to new camera
        new_pix_coords = K_rect @ new_cam_points[:, :3]
        new_pix_coords = new_pix_coords[:, :2] / (new_pix_coords[:, 2:3] + 1e-9)
        new_pix_coords = new_pix_coords.view_as(self.grid)

        return new_pix_coords

    def backproject_depth(self, depth, inv_K_rect):
        """
        Backproject depth maps to 3d points.
        """
        # initialize
        b, _, h, w = depth.shape
        if self.height != h or self.width != w:
            self.init_grid(b, h, w, depth.device, depth.dtype)
        cam_points = depth.view(b, 1, -1) * (inv_K_rect @ self.pix_coords)
        cam_points = torch.cat([cam_points, self.ones], dim=1)

        return cam_points

    def disparity_to_depth(self, disparity):
        """
        Sigmoid output of network to to depth.
        """
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disparity
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def get_transformation_matrix(self, axis_angle, translation):
        """
        Get 4x4 transformation matrix from axis-angle and translation.
        """
        # get rotation matrix
        rotation = self.rodrigues(axis_angle)

        # combine with translation into 4x4 transformation matrix
        b, _ = axis_angle.shape
        transformation = torch.eye(4, 4, device=axis_angle.device, dtype=axis_angle.dtype).repeat(b, 1, 1)
        transformation[:, :3, :3] = rotation
        transformation[:, :3, 3] = translation

        return transformation

    @staticmethod
    def rodrigues(axis_angle):
        """
        Convert axis-angle to rotation matrix using Rodrigues' formula.
        """
        # get norm
        angle = axis_angle.norm(2, dim=-1, keepdim=True)
        axis = axis_angle / (angle + 1e-6)

        # get skew-symmetric matrix
        b, _ = axis.shape
        axis_cross = torch.zeros(b, 3, 3, device=axis.device, dtype=axis.dtype)
        axis_cross[:, 0, 1] = -axis[:, 2]
        axis_cross[:, 0, 2] = axis[:, 1]
        axis_cross[:, 1, 0] = axis[:, 2]
        axis_cross[:, 1, 2] = -axis[:, 0]
        axis_cross[:, 2, 0] = -axis[:, 1]
        axis_cross[:, 2, 1] = axis[:, 0]

        # rodrigues formula for 3x3 rotation matrix
        eye = torch.eye(3, device=axis.device, dtype=axis.dtype)
        angle = angle.unsqueeze(-1)
        rotation = eye + torch.sin(angle) * axis_cross + (1 - torch.cos(angle)) * axis_cross @ axis_cross

        return rotation
