import io
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb

from .data_utils import batched
from .disparity import DisparityToFlow
from .visualizer_utils import disparity_map_to_image, event_frame_to_image, flow_map_to_image


class RerunVisualizer:
    """
    Live visualizer using Rerun.
    """

    def __init__(self, app_id, server, web, compression, blueprint=None):
        rr.init(app_id)
        rr.serve() if web else rr.connect(server)

        self.compression = compression
        self.counter = 0
        self.pose = (np.eye(3), np.zeros(3))
        self.linestrips = [np.zeros(3)]
        rr.log("pose", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        if blueprint is not None:
            self.blueprint = blueprint
            rr.send_blueprint(self.blueprint, make_active=True)

    def update_blueprint(self, quantities):
        if not hasattr(self, "blueprint"):
            # make views
            views = [rrb.Spatial2DView(name=q, origin=q) for q in quantities if q != "pose"]  # 2d
            views.append(rrb.Spatial3DView(name="pose", origin="pose")) if "pose" in quantities else None  # 3d

            # make into rows, then stack into blueprint and send it
            rows = batched(views, 3)
            self.blueprint = rrb.Blueprint(rrb.Vertical(*[rrb.Horizontal(*row) for row in rows]))
            rr.send_blueprint(self.blueprint, make_active=True)

    def set_counter(self):
        rr.set_time_sequence("frame", self.counter)
        self.counter += 1

    def event_frame(self, frame, name="events"):
        image = event_frame_to_image(frame)
        self.log_image(name, image, self.compression)

    def raw_map(self, frame, name="raw"):
        pass

    def flow_map(self, frame, name="flow"):
        image = flow_map_to_image(frame)
        self.log_image(name, image, self.compression)

    def disparity_map(self, frame, name="disparity"):
        image = disparity_map_to_image(frame)
        self.log_image(name, image, self.compression)

    def color_image(self, frame, name="color"):
        image = frame.permute(1, 2, 0).numpy().astype(np.uint8)
        self.log_image(name, image, self.compression)

    def pose_trajectory(self, pose, name="pose"):
        # https://rerun.io/docs/reference/types/archetypes/transform3d
        # NOTE: https://github.com/rerun-io/cpp-example-ros-bridge/blob/c65e24b8f85b05812df7288b79440658a96a7fcc/rerun_bridge/src/rerun_bridge/rerun_ros_interface.cpp#L69
        axis_angle, translation = pose.split([3, 3], dim=-1)
        rotation = DisparityToFlow.rodrigues(axis_angle.unsqueeze(0)).squeeze(0).numpy()
        translation = translation.numpy()

        orientation, origin = self.pose
        orientation = rotation @ orientation
        destination = origin + orientation @ translation
        self.pose = (orientation, destination)
        self.linestrips.append(destination)
        rr.log(name, rr.LineStrips3D(np.stack(self.linestrips), radii=0.001))
        rr.log(f"/{name}/translation/x", rr.Scalar(translation[0]))
        rr.log(f"/{name}/translation/y", rr.Scalar(translation[1]))
        rr.log(f"/{name}/translation/z", rr.Scalar(translation[2]))
        rr.log(f"/{name}/rotation/x", rr.Scalar(axis_angle[0]))
        rr.log(f"/{name}/rotation/y", rr.Scalar(axis_angle[1]))
        rr.log(f"/{name}/rotation/z", rr.Scalar(axis_angle[2]))

    @staticmethod
    def log_image(name, image_nd_array, compression=False):
        # compression could be None, "jpeg", "png"
        if compression:
            with io.BytesIO() as output:
                Image.fromarray(image_nd_array).save(output, format=compression)
                media_type = f"image/{compression.lower()}"
                rr.log(name, rr.EncodedImage(contents=output.getvalue(), media_type=media_type))
        else:
            rr.log(name, rr.Image(image_nd_array))

    @staticmethod
    def log_scalar(name, scalar):
        rr.log(name, rr.Scalar(scalar))

    @staticmethod
    def log_tensor(name, tensor):
        rr.log(name, rr.Tensor(tensor))


class ImageVisualizer:
    def __init__(self, root_dir, names, format):
        self.root_dir = Path(root_dir)
        self.format = format
        shutil.rmtree(self.root_dir) if self.root_dir.exists() else None
        for name in names:
            (self.root_dir / name).mkdir(exist_ok=True, parents=True)

        self.counter = 0

    def set_counter(self):
        self.counter += 1

    def save_image(self, name, image):
        if (self.root_dir / name).exists():
            image = Image.fromarray(image)
            image.save(self.root_dir / name / f"{self.counter:05d}.{self.format}")

    def save_nparray(self, frame, name):
        if (self.root_dir / name).exists():
            np.save(self.root_dir / name / f"{self.counter:05d}.npy", frame)

    def event_frame(self, frame, name="events"):
        image = event_frame_to_image(frame)
        self.save_image(name, image)

    def raw_map(self, frame, name="raw"):
        self.save_nparray(frame, name)

    def flow_map(self, frame, name="flow"):
        image = flow_map_to_image(frame)
        self.save_image(name, image)

    def disparity_map(self, frame, name="disparity"):
        image = disparity_map_to_image(frame)
        self.save_image(name, image)

    def color_image(self, frame, name="color"):
        image = frame.permute(1, 2, 0).numpy().astype(np.uint8)
        self.save_image(name, image)

    def scalar(self, name, scalar):
        with open(self.root_dir / name / "data.txt", "a") as f:
            f.write(f"{self.counter:05d},{scalar}\n")
