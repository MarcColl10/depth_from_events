import torch
import torch.nn as nn
import torch.nn.functional as F
import rerun as rr


class DepthDisparityToFlow(nn.Module):
    """
    Convert depth/disparity and pose to flow, optionally with learned intrinsics.

    Based on https://github.com/nianticlabs/monodepth2.
    Intrinsic matrix representation based on
    https://github.com/google-research/google-research/blob/master/depth_and_motion_learning/intrinsics_utils.py.
    """

    def __init__(self, mode, min_depth, max_depth, static=False):
        super().__init__()

        self.mode = mode
        self.backproject_depth = self.backproject_depth_static if static else self.backproject_depth_dynamic
        self.min_depth, self.max_depth = min_depth, max_depth
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
            inv_K_rect = torch.linalg.inv(K_rect)

        # split pose into axis-angle representation and translation
        axis_angle, translation = pose.split([3, 3], dim=-1)

        # transformation matrix (oldest -> most recent event ts)
        transformation = self.get_transformation_matrix(axis_angle, translation)

        # convert disparity to depth
        if self.mode == "disparity":
            _, depth = self.disparity_to_depth(disparity)
        else:
            # depth = self.clip_depth(disparity)
            depth = disparity

        # backproject depth maps to 3d points
        cam_points = self.backproject_depth(depth, inv_K_rect)

        # project 3d points to new cam location
        new_pix_coords = self.project_3d(cam_points, transformation, K_rect)

        # compute flow, convert -xy to xy
        flow = -(new_pix_coords - self.grid)

        return flow

    def clip_depth(self, depth):
        return depth.clamp(self.min_depth, self.max_depth)

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

    def backproject_depth_dynamic(self, depth, inv_K_rect):
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

    def backproject_depth_static(self, depth, inv_K_rect):
        b, _, _, _ = depth.shape
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

    def depth_to_disparity(self, depth):
        """
        Depth to sigmoid output of network.
        """
        # min_disp = 1 / self.max_depth
        # max_disp = 1 / self.min_depth
        # scaled_disp = 1 / depth
        # disparity = (scaled_disp - min_disp) / (max_disp - min_disp)
        return (1 / depth).nan_to_num(posinf=0)

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


class DepthDisparityMetrics(nn.Module):
    def __init__(self, metrics, scales, mask_by_events, cut_offs):
        super().__init__()

        self.accumulation_window = 1  # always backward
        self.metrics = metrics
        self.scales = scales
        # self.clamps = clamps
        self.mask_by_events = mask_by_events
        self.cut_offs = cut_offs

        self.results = dict()
        self.passes = 0

    def forward(self, event_frame, pred_map, gt_map):
        # check if gt is available
        if gt_map is None:
            return

        # or if we're doing evaluation without gt
        elif isinstance(gt_map, int):
            # compute differently scaled predictions
            assert any([isinstance(scale, (int, float)) for scale in self.scales]), "need scale for eval"
            for scale in self.scales:
                if isinstance(scale, (int, float)):
                    self.results[f"depth_disparity_eval_{scale}"] = (gt_map, pred_map * scale)

            self.passes += 1
            return

        # compute different gt masks
        gt_masks = dict()
        gt_nonzero = gt_map.ne(0)  # mask by valid gt
        for mask in self.mask_by_events:
            # mask by events
            gt_events = event_frame.sum(1).gt(0) if mask else torch.ones_like(gt_map, dtype=torch.bool)
            for cutoff in self.cut_offs:
                # mask by cutoff distance
                gt_cutoff = gt_map.abs() < cutoff if cutoff is not None else torch.ones_like(gt_map, dtype=torch.bool)
                gt_masks[f"events{mask}_cutoff{cutoff}"] = gt_nonzero * gt_events * gt_cutoff

        # compute differently scaled predictions
        scaled_pred_maps = dict()
        for scale in self.scales:
            if scale == "median":
                gt_median = gt_map[gt_nonzero].median()
                pred_median = pred_map[gt_nonzero].median()
                scaled_pred_maps[scale] = pred_map * (gt_median / pred_median)
                self.results["scale"] = (gt_median / pred_median).item()
            elif isinstance(scale, (int, float)):
                scaled_pred_maps[scale] = pred_map * scale
            self.results[f"depth_disparity_eval_{scale}"] = (gt_map.clone(), scaled_pred_maps[scale].clone())
            # per pixel histogram
            max_disparity = torch.max(torch.max(gt_map), torch.max(scaled_pred_maps[scale]))
            bins = 30
            pred_hist = torch.histogram(scaled_pred_maps[scale].cpu(), bins=bins, range=(0, max_disparity))
            gt_hist = torch.histogram(gt_map.cpu(), bins=bins, range=(0, max_disparity))
            self.results[f"hist_{scale}"] = (gt_hist, pred_hist)

        # go over all metrics
        for metric in self.metrics:

            # mae
            if metric == "mae":
                # go over all scales
                for scale, map_ in scaled_pred_maps.items():
                    # go over all masks
                    for mask_name, mask in gt_masks.items():
                        # depth_map = depth_map.clamp(*clamp)
                        mae = F.l1_loss(map_[mask], gt_map[mask], reduction="mean")
                        # for visualization only
                        map_v_ = map_.clone()
                        gt_map_v_ = gt_map.clone()
                        map_v_[~mask] = 0
                        gt_map_v_[~mask] = 0
                        # ad-hoc fix so that the saved disparity does not get plotted as events
                        mask_name = mask_name.replace("events", "event")
                        self.results[f"depth_disparity_eval_{scale}_{mask_name}"] = (gt_map_v_, map_v_)
                        self.results[f"{metric}_{scale}_{mask_name}"] = mae.item()

                        # per pixel histogram
                        max_disparity = torch.max(torch.max(gt_map[mask]), torch.max(map_[mask]))
                        bins = 30
                        pred_hist = torch.histogram(map_[mask].cpu(), bins=bins, range=(0, max_disparity))
                        gt_hist = torch.histogram(gt_map[mask].cpu(), bins=bins, range=(0, max_disparity))
                        self.results[f"hist_{scale}_{mask_name}"] = (gt_hist, pred_hist)

            if metric == "absrel":
                for scale, map_ in scaled_pred_maps.items():
                    for mask_name, mask in gt_masks.items():
                        absrel = torch.abs(map_[mask] - gt_map[mask]) / gt_map[mask]
                        # absrel = absrel[~torch.isnan(absrel)]
                        # absrel = absrel[~torch.isinf(absrel)]
                        absrel = absrel.mean()
                        mask_name = mask_name.replace("events", "event")
                        self.results[f"{metric}_{scale}_{mask_name}"] = absrel.item()

            if metric == "rmselog":
                for scale, map_ in scaled_pred_maps.items():
                    for mask_name, mask in gt_masks.items():
                        rmselog = torch.sqrt(F.mse_loss(torch.log(map_[mask] + 1e-9), torch.log(gt_map[mask] + 1e-9)))
                        mask_name = mask_name.replace("events", "event")
                        self.results[f"{metric}_{scale}_{mask_name}"] = rmselog.item()

            if metric == "silog":
                for scale, map_ in scaled_pred_maps.items():
                    for mask_name, mask in gt_masks.items():
                        diff = torch.log(map_[mask] + 1e-9) - torch.log(gt_map[mask] + 1e-9)
                        silog = torch.mean(diff**2) - (torch.mean(diff) ** 2)
                        mask_name = mask_name.replace("events", "event")
                        self.results[f"{metric}_{scale}_{mask_name}"] = silog.item()

        self.passes += 1

    def backward(self):
        pass

    def reset(self):
        self.results = dict()
        self.passes = 0

    def compute_and_reset(self):
        results = self.results.copy()
        self.reset()
        return results
