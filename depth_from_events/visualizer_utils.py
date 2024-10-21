import matplotlib.cm as cm
from matplotlib.colors import hsv_to_rgb, Normalize
import numpy as np


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
    disp_valid = 1 / disparity[y, x]
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
