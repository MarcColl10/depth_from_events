from collections import deque

import torch
import torch.nn as nn


def recursive_detach(x):
    """
    Detach all tensors in a nested structure.
    """
    if isinstance(x, torch.Tensor):
        return x.detach()
    elif isinstance(x, tuple):
        return tuple(recursive_detach(xx) for xx in x)
    elif isinstance(x, deque):
        return deque([recursive_detach(xx) for xx in x], maxlen=x.maxlen)
    elif x is None:
        return None
    else:
        raise ValueError(f"Unknown type {type(x)}")


class NetworkWrapper(nn.Module):
    """
    Wrapper for ease of use during training, while allowing the base
    network to be compiled with e.g. TensorRT.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reset_state()

    def forward(self, input):
        hidden = self.get_state()
        x, hidden = super().forward(input, hidden)
        self.set_state(hidden)
        return x

    def trace(self, input, device="cpu"):
        with torch.no_grad():
            self(input.to(device))
        self.reset()

    def reset(self):
        self.reset_state()

    def detach(self):
        self.detach_state()

    def reset_state(self):
        self.state = None

    def detach_state(self):
        self.state = recursive_detach(self.state)

    def get_state(self):
        return self.state[0] if self.state is not None else None

    def set_state(self, *state):
        self.state = state
