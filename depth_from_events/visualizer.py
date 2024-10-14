import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
import io

from .data_utils import batched
from .disparity import DisparityToFlow
from .visualizer_utils import disparity_map_to_image, event_frame_to_image, flow_map_to_image


class RerunVisualizer:
    """
    Live visualizer using Rerun.
    """

    def __init__(self, app_id, server, web):
        rr.init(app_id)
        rr.serve() if web else rr.connect(server)

        self.counter = 0
        self.pose = (np.eye(3), np.zeros(3))
        self.linestrips = [np.zeros(3)]
        rr.log("pose", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

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
        self.log_image(name, image, "jpeg")

    def flow_map(self, frame, name="flow"):
        image = flow_map_to_image(frame)
        self.log_image(name, image, "jpeg")

    def disparity_map(self, frame, name="disparity"):
        image = disparity_map_to_image(frame)
        self.log_image(name, image, "jpeg")

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
