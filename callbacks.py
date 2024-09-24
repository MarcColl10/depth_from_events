from lightning.pytorch.callbacks import Callback
import numpy as np

from visualizer import RerunVisualizer, flow_map_to_image


class LiveVisualizer(Callback):
    def __init__(self, app_id, server, web):
        self.visualizer = RerunVisualizer(app_id, server, web)

    def on_batch_end(self, outputs):
        for events, flow in zip(outputs.frame, outputs.flow):
            self.visualizer.set_counter()
            self.visualizer.event_frame(events[0].cpu())
            self.visualizer.flow_map(flow[0].detach().cpu())

    def on_train_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)

    def on_validation_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)


# TODO: WIP
class ValidationVideoLogger(Callback):
    def __init__(self, recording, range):
        self.recording = recording
        self.range = range
        self.counter = None

    def on_validation_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        # init or reset counter
        if batch.recording == self.recording and self.counter is None:
            self.counter = 0
        elif batch.recording != self.recording and self.counter is not None:
            self.counter = None

        # log video if range inside received logs
        if batch.recording == self.recording:
            start, end = self.range
            if start >= self.counter and end < self.counter + len(outputs.flow):  # fully inside
                flow_slice = [f[0].detach().cpu() for f in outputs.flow[start - self.counter : end - self.counter]]
                flow_images = np.stack([flow_map_to_image(flow_map) for flow_map in flow_slice]).transpose(0, 1, 3, 2)
                # https://docs.wandb.ai/guides/track/log/media/#other-media
                litmodule.logger.experiment.log({f"flow_{self.recording}_{start}_{end}": flow_images})
            self.counter += len(outputs.flow)
