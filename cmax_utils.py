import torch
from torch.nn.utils.rnn import pad_sequence


def extract_events_from_frames(frames):
    """
    Extract 'events' from frames whose pixels have channels
    (neg polarity count, pos polarity count, avg quantized ts).
    """
    # get shape
    b, _, _, _, _ = frames.shape

    # do per-polarity
    # polarity channel is nonzero
    output = [[] for _ in range(b)]
    nonzero_mask = frames[:, :2].gt(0)  # (b, p, d, h, w)
    nonzero_indices = nonzero_mask.nonzero(as_tuple=True)
    bi, pi, zi, yi, xi = nonzero_indices

    # combine nonzero polarities with xy and ts coordinates
    avg_ts = frames[bi, -1, zi, yi, xi] + zi  # increment by passes
    xyz = torch.stack([xi, yi, avg_ts, zi], dim=1)
    pol = frames[bi, pi, zi, yi, xi] * (2 * pi - 1)
    combined = torch.cat([xyz, pol.view(-1, 1)], dim=1)

    # append
    for i in range(b):
        output[i] += [combined[bi == i]]

    # pad to batch
    output = pad_sequence([torch.cat(o) for o in output], batch_first=True)

    return output


def format_events(events, counts):
    """
    Go from list of padded (b, n, 4) events with (t, y, x, p)
    to padded (b, n, 5) events with (x, y, t, pass, p).
    """
    max_counts = [c.max() for c in counts]  # per batch
    output = []
    for i, (ev, c) in enumerate(zip(events, max_counts)):
        t, y, x, p = ev[:, :c].unbind(-1)
        z = torch.ones_like(t) * i
        output.append(torch.stack([x, y, t + i, z, p], dim=-1))
    output = torch.cat(output, dim=1)
    return output
