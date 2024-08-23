import torch
import torch.nn as nn

from cmax_utils import extract_events_from_frames
from cuda_3d_ops import iterative_3d_warp_cuda
from iwe import build_iwe


class IterativeContrastMaximization(nn.Module):

    def __init__(self, accumulation_window, base):
        super().__init__()

        self.accumulation_window = accumulation_window
        self.base = base

        self.total_loss = 0
        self.passes = 0
        self.event_frames = []
        self.flow_maps = []

    def forward(self, event_frame, flow_map):
        # store
        self.event_frames.append(event_frame)
        self.flow_maps.append(flow_map)
        self.passes += 1

    def backward(self):
        # extract events
        event_frames = torch.stack(self.event_frames, dim=2)  # (b, c, d, h, w)
        events = extract_events_from_frames(event_frames)  # padded (b, n, 4)

        # if no events, no loss
        # TODO: for some reason catching 0 dim in cuda doesn't work
        _, n, _ = events.shape
        if not n:
            return self.total_loss

        # stack flows
        flow_maps = torch.stack(self.flow_maps, dim=2)  # (b, 2, d, h, w)
        flow_maps = flow_maps.permute(0, 2, 3, 4, 1).contiguous()  # (b, d, h, w, 2)

        # warp events: (b, n, 4) -> (b, n, d + 1, 5) with (x, y, t, t_orig, p)
        warped_events = iterative_3d_warp_cuda(events, flow_maps)

        # build iwe and iwt with (trilinear) splatting
        _, _, _, h, w = event_frames.shape
        iwe, iwt = build_iwe(warped_events, self.base, (h, w))  # (b, 2, d + 1, h, w)

        # split into negative and positive polarity
        iwe_neg, iwe_pos = iwe.unbind(1)
        iwt_neg, iwt_pos = iwt.unbind(1)

        # per-polarity image of warped average timestamps
        iwat_neg = iwt_neg / (iwe_neg + 1e-9)
        iwat_pos = iwt_pos / (iwe_pos + 1e-9)

        # scale by number of pixels with at least one event in iwe
        inside = (iwe_neg + iwe_pos).gt(0).float().flatten(start_dim=2).sum(2) + 1e-9

        # compute deblurring loss
        # mean over batch and reference times
        loss = iwat_neg.pow(2).flatten(start_dim=2).sum(2) + iwat_pos.pow(2).flatten(start_dim=2).sum(2)
        self.total_loss += (loss / inside).mean()

        return self.total_loss

    def reset(self):
        self.total_loss = 0
        self.passes = 0
        self.event_frames.clear()
        self.flow_maps.clear()

    def compute_and_reset(self):
        mean_loss = self.total_loss  # already mean over passes
        self.reset()
        return {"cmax": mean_loss}
