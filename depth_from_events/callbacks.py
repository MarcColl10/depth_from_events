from lightning.pytorch.callbacks import Callback

from .visualizer import ImageVisualizer, RerunVisualizer


class LiveVisualizer(Callback):
    def __init__(self, app_id, server, web, blueprint=None):
        self.visualizer = RerunVisualizer(app_id, server, web, blueprint)

    def on_batch_end(self, outputs):
        # update blueprint
        all_keys = set()
        [all_keys.update(output.keys()) for output in outputs.values()]
        self.visualizer.update_blueprint(list(all_keys))

        for output in outputs.values():
            self.visualizer.set_counter()

            # things with events
            for k in [k for k in output.keys() if "events" in k]:
                self.visualizer.event_frame(output[k][0].detach().cpu(), name=k)

            # things with flow
            for k in [k for k in output.keys() if k.endswith("flow")]:
                self.visualizer.flow_map(output[k][0].detach().cpu(), name=k)

            # things with disparity
            for k in [k for k in output.keys() if "disparity" in k]:
                self.visualizer.disparity_map(output[k][0].detach().cpu(), name=k)

            # things with pose
            for k in [k for k in output.keys() if "pose" in k]:
                self.visualizer.pose_trajectory(output[k][0].detach().cpu(), name=k)

            # for scalar values
            for k in [k for k in output.keys() if isinstance(output[k], (int, float))]:
                self.visualizer.log_scalar(k, output[k])

    def on_train_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)

    def on_validation_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)


class ImageLogger(Callback):
    def __init__(self, root_dir, keys):
        self.visualizer = ImageVisualizer(root_dir, keys)

    def on_batch_end(self, outputs):
        for output in outputs.values():
            self.visualizer.set_counter()

            # things with events
            for k in [k for k in output.keys() if "events" in k]:
                self.visualizer.event_frame(output[k][0].detach().cpu(), name=k)

            # things with flow
            for k in [k for k in output.keys() if k.endswith("flow")]:
                self.visualizer.flow_map(output[k][0].detach().cpu(), name=k)

            # things with disparity
            for k in [k for k in output.keys() if "disparity" in k]:
                self.visualizer.disparity_map(output[k][0].detach().cpu(), name=k)

    def on_train_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)

    def on_validation_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)
