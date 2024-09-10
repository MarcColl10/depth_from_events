from lightning.pytorch.callbacks import Callback

from visualizer import RerunVisualizer


class LiveVisualizer(Callback):
    def __init__(self, name):
        self.visualizer = RerunVisualizer(name)

    def on_batch_end(self, outputs):
        for events, flow in zip(outputs.frame, outputs.flow):
            self.visualizer.set_counter()
            self.visualizer.event_frame(events[0].cpu())
            self.visualizer.flow_map(flow[0].detach().cpu())

    def on_train_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)

    def on_validation_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)
