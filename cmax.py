import torch
import torch.nn as nn

from cmax_utils import extract_events_from_frames, format_events
from cuda_3d_ops import iterative_3d_warp_cuda
from iwe import build_iwe


class ContrastMaximization(nn.Module):
    """
    Base class.
    """

    cls_name = None

    def __init__(self, use_events, accumulation_window, base, keep_warping, select):
        super().__init__()

        self.use_events = use_events
        self.accumulation_window = accumulation_window
        self.base = base
        self.keep_warping = keep_warping
        self.select = select

        self.total_loss = 0
        self.passes = 0
        self.event_frames = []
        self.auxs = []
        self.flow_maps = []

    def forward(self, event_frame, aux, flow_map):
        self.event_frames.append(event_frame)
        self.auxs.append(aux)
        self.flow_maps.append(flow_map)
        self.passes += 1

    def prepare_backward(self):
        # extract events
        if not self.use_events:
            event_frames = torch.stack(self.event_frames, dim=2)  # (b, c, d, h, w)
            events = extract_events_from_frames(event_frames)  # padded (b, n, 5)
        else:
            counts = [aux.counts for aux in self.auxs]
            events = [aux.events for aux in self.auxs]
            events = format_events(events, counts)  # padded (b, n, 5)

        # stack flows
        flow_maps = torch.stack(self.flow_maps, dim=2)  # (b, 2, d, h, w)
        flow_maps = flow_maps.permute(0, 2, 3, 4, 1).contiguous()  # (b, d, h, w, 2)

        return events, flow_maps

    def compute_cmax_loss(self, events, flow_maps):
        # warp events: (b, n, 5) -> (b, n, d + 1, 5) with (x, y, t, t_orig, p)
        warped_events = iterative_3d_warp_cuda(events, flow_maps, self.base, self.keep_warping)

        # build iwe and iwt with (trilinear) splatting
        _, _, h, w, _ = flow_maps.shape
        iwe, iwt = build_iwe(warped_events, self.base, self.select, (h, w))  # (b, 2, d + 1, h, w)

        # split into negative and positive polarity
        iwe_neg, iwe_pos = iwe.unbind(1)
        iwt_neg, iwt_pos = iwt.unbind(1)

        # per-polarity image of warped average timestamps
        iwat_neg = iwt_neg / (iwe_neg + 1e-9)
        iwat_pos = iwt_pos / (iwe_pos + 1e-9)

        # scale by number of pixels with at least one event in iwe
        inside = (iwe_neg + iwe_pos).gt(0).float().flatten(start_dim=2).sum(2) + 1e-9

        # compute deblurring loss
        loss = iwat_neg.pow(2).flatten(start_dim=2).sum(2) + iwat_pos.pow(2).flatten(start_dim=2).sum(2)
        loss = loss / inside
        return loss

    def get_accumulated_events(self):
        if not self.use_events:
            accumulated_event_frame = torch.stack(self.event_frames, dim=2).sum(2)
            return accumulated_event_frame
        else:
            counts = [aux.counts for aux in self.auxs]
            events = [aux.events for aux in self.auxs]
            accumulated_events = format_events(events, counts, stack=True)  # padded (b, n, d, 5)
            accumulated_events[..., 2] = accumulated_events[..., 2].floor()  # prevent trilinear splat
            if not accumulated_events.numel():
                return torch.zeros_like(self.event_frames[0])
            _, _, h, w = self.event_frames[0].shape
            accumulated_event_frame, _ = build_iwe(accumulated_events, 1, None, (h, w))  # (b, 2, d, h, w)
            return accumulated_event_frame.sum(2)

    def get_accumulated_flow(self, tref):
        # TODO: use extract_events?
        pass

    def compute_iwe(self, tref):
        # get events and flow maps
        events, flow_maps = self.prepare_backward()
        if not events.numel():
            return torch.zeros_like(self.event_frames[0])

        # warp events: (b, n, 5) -> (b, n, d + 1, 5) with (x, y, t, t_orig, p)
        warped_events = iterative_3d_warp_cuda(events, flow_maps, self.base, self.keep_warping)

        # build iwe and iwt with (trilinear) splatting
        _, _, h, w, _ = flow_maps.shape
        iwe, _ = build_iwe(warped_events, self.base, self.select, (h, w))  # (b, 2, d + 1, h, w)
        return iwe[:, :, tref]

    def backward(self):
        raise NotImplementedError

    def reset(self):
        self.total_loss = 0
        self.passes = 0
        self.event_frames.clear()
        self.auxs.clear()
        self.flow_maps.clear()

    def compute_and_reset(self):
        mean_loss = self.total_loss  # already mean over passes
        self.reset()
        return {self.cls_name: mean_loss}


class IterativeContrastMaximization(ContrastMaximization):
    """
    Contrast maximization loss as used in Paredes-Valles et al., ICCV 2023.
    Involves iterative warping to multiple references (all bin edges in deblurring window).
    Contrast maximization is done on the warped image of average timestamps.
    """

    cls_name = "cmax"

    def backward(self):
        # get events and flow maps
        events, flow_maps = self.prepare_backward()

        # if no events, no loss
        # TODO: for some reason catching 0 dim in cuda doesn't work
        if not events.numel():
            return None

        # compute deblurring loss
        # mean over batch and reference times
        loss = self.compute_cmax_loss(events, flow_maps)
        self.total_loss += loss.mean()

        return self.total_loss


class RatioSquaredAvgTimestamps(ContrastMaximization):
    """
    RSAT loss from Hagenaars and Paredes-Valles et al., NeurIPS 2021.
    Quantifies how much warping by optical flow increases sharpness of the image of (warped) events.
    """

    cls_name = "rsat"

    def backward(self):
        # get events and flow maps
        events, flow_maps = self.prepare_backward()

        # if no events, no loss
        if not events.numel():
            return None

        # get zero flow
        zero_maps = torch.zeros_like(flow_maps)

        # compute deblurring loss
        # mean over batch, only last reference time
        loss = self.compute_cmax_loss(events, flow_maps)
        loss_unwarped = self.compute_cmax_loss(events, zero_maps)
        rsat = loss[:, -1] / (loss_unwarped[:, -1] + 1e-9)
        self.total_loss += rsat.mean()

        return self.total_loss
