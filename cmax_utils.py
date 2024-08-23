import torch


def extract_events_from_frames(frames):
    """
    Extract 'events' from frames whose pixels have channels
    (neg polarity count, pos polarity count, avg quantized ts).
    """
    # get shape
    b, _, _, _, _ = frames.shape

    # do per-polarity
    output = [[] for _ in range(b)]
    for pi, factor in zip([0, 1], [-1, 1]):
        # polarity channel is nonzero
        nonzero_mask = frames[:, pi].gt(0)  # (b, d, h, w)
        nonzero_indices = nonzero_mask.nonzero(as_tuple=True)
        bi, zi, yi, xi = nonzero_indices

        # combine nonzero polarities with xy and ts coordinates
        avg_ts = frames[bi, -1, zi, yi, xi] + zi  # increment by passes
        xyz = torch.stack([xi, yi, avg_ts, zi], dim=1)
        pol = frames[bi, pi, zi, yi, xi].view(-1, 1) * factor
        combined = torch.cat([xyz, pol], dim=1)

        # append
        for i in range(b):
            output[i] += [combined[bi == i]]

    # pad to batch
    output = torch.nn.utils.rnn.pad_sequence([torch.cat(o) for o in output], batch_first=True)

    return output
