import torch
import torch.nn as nn


class EdgeAwareSmoothing(nn.Module):
    """
    Smoothing loss that is tuned down in areas with high event image gradients.
    Implementation from https://github.com/nianticlabs/monodepth2.
    """

    def __init__(self, accumulation_window, weight):
        super().__init__()

        self.accumulation_window = accumulation_window
        self.weight = weight

        self.total_loss = 0
        self.passes = 0
        self.event_frames = []
        self.pred_maps = []

    def forward(self, event_frame, pred_map):
        self.event_frames.append(event_frame)
        self.pred_maps.append(pred_map)
        self.passes += 1

    def backward(self):
        # for all steps
        for i, (event_frame, pred_map) in enumerate(zip(self.event_frames, self.pred_maps)):
            # compute image gradients, mean over polarities
            img_grad_y = (event_frame[..., :-1, :] - event_frame[..., 1:, :]).abs().mean(1, keepdim=True)
            img_grad_x = (event_frame[..., :-1] - event_frame[..., 1:]).abs().mean(1, keepdim=True)

            # normalize prediction map
            norm_pred_map = pred_map / (pred_map.mean((2, 3), keepdim=True) + 1e-9)

            # compute prediction gradients
            pred_grad_y = (norm_pred_map[..., :-1, :] - norm_pred_map[..., 1:, :]).abs()
            pred_grad_x = (norm_pred_map[..., :-1] - norm_pred_map[..., 1:]).abs()

            # edge-aware smoothing loss
            pred_grad_y *= torch.exp(-img_grad_y)
            pred_grad_x *= torch.exp(-img_grad_x)
            self.total_loss += pred_grad_y.mean() + pred_grad_x.mean()

            # temporal component
            if i + 1 < self.passes:
                next_event_frame = self.event_frames[i + 1][:, :2]
                img_grad_t = (event_frame - next_event_frame).abs().mean(1, keepdim=True)

                next_pred_map = self.pred_maps[i + 1]
                next_norm_pred_map = next_pred_map / (next_pred_map.mean((2, 3), keepdim=True) + 1e-9)
                pred_grad_t = (norm_pred_map - next_norm_pred_map).abs()

                pred_grad_t *= torch.exp(-img_grad_t)
                self.total_loss += pred_grad_t.mean()

        self.total_loss /= self.passes
        self.total_loss *= self.weight

        return self.total_loss

    def reset(self):
        self.total_loss = 0
        self.passes = 0
        self.event_frames.clear()
        self.pred_maps.clear()

    def compute_and_reset(self):
        mean_loss = self.total_loss
        self.reset()
        return {"ea_smooth": mean_loss}
