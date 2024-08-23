from lightning.pytorch.callbacks import Callback
import torch

from visualizer import RerunVisualizer


class Trace(Callback):
    """
    We use lazy PyTorch modules, so we need to initialize the parameters
    by tracing them with data. Batch size doesn't matter as long as it doesn't
    influence the parameter shapes.

    NOTE: callbacks should contain optional code, this is in litmodule setup now
    """

    def trace(self, network, device):
        x = torch.zeros(1, 2, 128, 128, device=device)
        with torch.no_grad():
            network(x)
            network.reset()

    def on_train_start(self, trainer, litmodule):
        device = "cuda" if trainer.accelerator == "gpu" else "cpu"
        self.trace(litmodule.network, device)

    def on_validation_start(self, trainer, litmodule):
        device = "cuda" if trainer.accelerator == "gpu" else "cpu"
        self.trace(litmodule.network, device)


class LiveVisualizer(Callback):
    def __init__(self, name):
        self.visualizer = RerunVisualizer(name)

    def on_train_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        for frame in batch.frames:
            self.visualizer.step(frame)
