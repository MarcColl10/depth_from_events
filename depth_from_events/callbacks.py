from pathlib import Path
import shutil

import cv2
from lightning.pytorch.callbacks import Callback
import numpy as np

from depth_from_events.visualizer import ImageVisualizer, RerunVisualizer


class LiveVisualizer(Callback):
    def __init__(self, app_id, server, web, compression, blueprint=None):
        self.visualizer = RerunVisualizer(app_id, server, web, compression, blueprint)

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
            for k in [k for k in output.keys() if "flow" in k and "raw" not in k]:
                self.visualizer.flow_map(output[k][0].detach().cpu(), name=k)

            # things with disparity
            for k in [k for k in output.keys() if "disparity" in k and "raw" not in k]:
                if isinstance(output[k], tuple):
                    self.visualizer.disparity_map(output[k][1][0].detach().cpu(), name=k)
                    self.visualizer.disparity_map(output[k][0][0].detach().cpu(), name=f"gt_{k}")
                else:
                    self.visualizer.disparity_map(output[k][0].detach().cpu(), name=k)

            # things with color
            for k in [k for k in output.keys() if "color" in k]:
                self.visualizer.color_image(output[k][0].detach().cpu(), name=k)

            # things with pose
            for k in [k for k in output.keys() if "pose" in k]:
                self.visualizer.pose_trajectory(output[k][0].detach().cpu(), name=k)

            # for scalar values
            for k in [k for k in output.keys() if isinstance(output[k], (int, float))]:
                self.visualizer.log_scalar(k, output[k])

            # for histograms
            for k in [k for k in output.keys() if k.startswith("hist")]:
                self.visualizer.log_tensor(f"{k}_gt", output[k][0].hist)
                self.visualizer.log_tensor(k, output[k][1].hist)

    def on_train_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)

    def on_validation_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)

    def on_test_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)


class ImageLogger(Callback):
    def __init__(self, root_dir, keys, format):
        self.visualizer = ImageVisualizer(root_dir, keys, format)

    def on_batch_end(self, outputs):
        for output in outputs.values():
            self.visualizer.set_counter()

            # things with events
            for k in [k for k in output.keys() if "events" in k]:
                self.visualizer.event_frame(output[k][0].detach().cpu(), name=k)

            # save as raw numpy array
            for k in [k for k in output.keys() if "raw" in k]:
                self.visualizer.nparray(output[k][0].detach().cpu().numpy(), name=k)

            # things with flow
            for k in [k for k in output.keys() if "flow" in k and "raw" not in k]:
                self.visualizer.flow_map(output[k][0].detach().cpu(), name=k)

            # things with disparity
            for k in [k for k in output.keys() if "disparity" in k and "raw" not in k]:
                self.visualizer.disparity_map(output[k][0].detach().cpu(), name=k)

            # color images
            for k in [k for k in output.keys() if "color" in k]:
                self.visualizer.color_image(output[k][0].detach().cpu(), name=k)

            # for scalar values
            for k in [k for k in output.keys() if isinstance(output[k], (int, float))]:
                self.visualizer.scalar(k, output[k])

    def on_train_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)

    def on_validation_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)

    def on_test_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        self.on_batch_end(outputs)


class StoreDsecEvalDisparity(Callback):
    """
    Write DSEC evaluation results to the file structure given in https://dsec.ifi.uzh.ch/disparity-submission-format/.
    """

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_test_batch_end(self, trainer, litmodule, outputs, batch, batch_idx):
        for output in outputs.values():

            disparity_keys = [key for key in output.keys() if key.startswith("depth_disparity")]

            for key in disparity_keys:
                rec = batch.recording
                eval_id, eval_disparity = output[key]
                eval_disparity = eval_disparity.cpu().numpy()

                # format following https://dsec.ifi.uzh.ch/disparity-submission-format/
                disp = eval_disparity.astype(np.float64).squeeze((0, 1))  # remove batch and channel dim
                formatted_disp = (disp * 256).astype(np.uint16)

                # write to file
                (self.output_dir / key / rec).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.output_dir / key / rec / f"{eval_id:06d}.png"), formatted_disp)

    def on_test_epoch_end(self, trainer, litmodule):
        shutil.make_archive(self.output_dir, "zip", self.output_dir)
