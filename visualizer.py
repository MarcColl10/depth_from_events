import matplotlib.cm as cm
from matplotlib.colors import hsv_to_rgb, Normalize
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
import io

from data_utils import batched
from disparity import DisparityToFlow


def event_frame_to_image(frame, pol_channels=[0, 1]):
    """
    Convert an event frame to an RGB image, where the red channel
    represents negative events and the blue channel positive events.

    Args:
        frame (np.ndarray): Event frame with shape (?, height, width).
        pol_channels (list): Indices of the negative and positive polarity channels.

    Returns:
        np.ndarray: RGB image of the event frame.
    """

    # event frame (c, h, w), with channels neg and pos polarity
    _, h, w = frame.shape
    frame = frame[pol_channels]

    # normalize per channel
    almost_max = np.percentile(frame, 99, axis=(1, 2), keepdims=True)
    almost_min = np.percentile(frame, 1, axis=(1, 2), keepdims=True)
    if (almost_min != almost_max).all():
        frame_norm = (frame - almost_min) / (almost_max - almost_min)
    else:
        frame_norm = frame
    frame_norm = np.clip(frame_norm, 0, 1)

    # write to rgb image of ints
    frame_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    neg, pos = frame_norm
    frame_rgb[:, :, 0] = neg * 255  # red
    frame_rgb[:, :, 2] = pos * 255  # blue

    return frame_rgb


def flow_map_to_image(frame):
    """
    Convert an optical flow map to an RGB image.

    Args:
        frame (np.ndarray): Optical flow map with shape (2, height, width) and (x, y) flow channels.

    Returns:
        np.ndarray: RGB image of the optical flow frame.
    """

    # check shape
    assert frame.ndim == 3 and frame.shape[0] == 2, "Flow must have shape (2, height, width)."

    # flow magnitude
    mag = (frame**2).sum(0) ** 0.5
    min_mag = mag.min()
    d_mag = mag.max() - min_mag

    # flow angle
    x, y = frame
    ang = np.arctan2(y, x) + np.pi
    ang *= 1.0 / np.pi / 2.0

    # flow color
    frame_hsv = np.stack([ang, np.ones_like(ang), mag - min_mag], axis=2)
    frame_hsv[:, :, 2] /= d_mag if d_mag != 0.0 else 1.0

    # to rgb ints
    frame_rgb = hsv_to_rgb(frame_hsv)
    frame_rgb = (frame_rgb * 255).astype(np.uint8)

    return frame_rgb


def disparity_map_to_image(disparity, reverse=False):
    """
    Convert a disparity (or depth) map to an RGB image.
    Source: https://github.com/uzh-rpg/DSEC/blob/main/scripts/dataset/visualization.py

    Args:
        disparity (np.ndarray): Disparity map with shape (1, height, width).
        reverse (bool): Whether to reverse the colormap (for depth maps).

    Returns:
        np.ndarray: RGB image of the disparity map.
    """

    # check shape
    assert disparity.ndim == 3 and disparity.shape[0] == 1, "Disparity must have shape (1, height, width)."

    # disparity magnitude for nonzero pixels
    disparity = disparity.squeeze(0)  # remove channel
    disp_pixels = np.argwhere(disparity > 0)
    y, x = disp_pixels
    disp_valid = disparity[y, x]
    min_disp = disp_valid.min() if len(disp_valid) > 0 else 0
    max_disp = disp_valid.max() if len(disp_valid) > 0 else 0

    # disparity colormap (in reverse if depth map)
    norm = Normalize(vmin=min_disp, vmax=max_disp, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap="inferno" if not reverse else "inferno_r")
    disp_color = mapper.to_rgba(disp_valid)[..., :3]
    image = np.zeros((*disparity.shape, 3))

    # to rgb ints
    image[y, x] = disp_color
    image = (image * 255).astype(np.uint8)

    return image


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
