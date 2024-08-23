import torch

from cuda_3d_ops import trilinear_splat_cuda


def build_iwe(warped_events, base, resolution):
    # get deblurring window
    b, _, d, _ = warped_events.shape

    # per-polarity image of warped events
    # TODO: make more efficient?
    x, y, t, t_orig, p = warped_events.unbind(-1)
    neg, pos = p.lt(0), p.gt(0)
    neg_warped_events = torch.stack([x * neg, y * neg, t * neg, (p * neg).abs()], dim=-1)
    pos_warped_events = torch.stack([x * pos, y * pos, t * pos, (p * pos).abs()], dim=-1)

    iwe_neg = trilinear_splat_cuda(neg_warped_events.view(b, -1, 4), (d, *resolution))
    iwe_pos = trilinear_splat_cuda(pos_warped_events.view(b, -1, 4), (d, *resolution))
    iwe = torch.stack([iwe_neg, iwe_pos], dim=1)  # (b, 2, d, h, w)

    # per-polarity image of warped timestamps
    t_ref = torch.arange(d, device=warped_events.device)
    t_contrib = 1 - (t_ref - t_orig).abs() / base
    neg_warped_events = torch.stack([x * neg, y * neg, t * neg, (p * neg).abs() * t_contrib], dim=-1)
    pos_warped_events = torch.stack([x * pos, y * pos, t * pos, (p * pos).abs() * t_contrib], dim=-1)

    iwt_neg = trilinear_splat_cuda(neg_warped_events.view(b, -1, 4), (d, *resolution))
    iwt_pos = trilinear_splat_cuda(pos_warped_events.view(b, -1, 4), (d, *resolution))
    iwt = torch.stack([iwt_neg, iwt_pos], dim=1)

    return iwe, iwt
