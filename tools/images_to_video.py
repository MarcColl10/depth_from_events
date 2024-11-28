import argparse
import os

import cv2
from moviepy.editor import ImageSequenceClip
import numpy as np
import pandas as pd


def create_video_with_yaw_rate(image_folder, output_video, status_file, yaw_rate_file, frame_range, scale_factor, fps):
    # load status data
    if status_file is not None:
        status_data = pd.read_csv(status_file, header=None, names=["frame", "status"])
        status_data = status_data.set_index("frame").to_dict()["status"]

    # load yaw rate data
    if yaw_rate_file is not None:
        yaw_data = pd.read_csv(yaw_rate_file, header=None, names=["frame", "yaw_rate"])
        yaw_data = yaw_data.set_index("frame").to_dict()["yaw_rate"]

    # get all image files, sorted numerically
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    image_indices = [int(os.path.splitext(f)[0]) for f in image_files]
    if frame_range is None:
        start = 0
        stop = max(image_indices) + 1
    else:
        start, stop = frame_range

    # load images and handle missing frames
    frames = []
    last_frame = None
    last_status, last_yaw_rate = False, 0
    for frame_idx in range(start, stop):
        if frame_idx in image_indices:
            img_path = os.path.join(image_folder, f"{frame_idx:05d}.png")
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB for moviepy
            last_frame = frame
        else:
            if last_frame is None:
                continue
            frame = last_frame  # repeat last frame if missing

        # draw yaw rate arrow
        if frame is not None and yaw_rate_file is not None:
            yaw_rate = yaw_data.get(frame_idx, last_yaw_rate)
            yaw_rate = np.clip(yaw_rate, -0.5, 0.5)  # clamp to -0.5 to 0.5
            last_yaw_rate = yaw_rate
            frame = draw_arrow_on_frame(frame, yaw_rate, scale_factor)
        else:
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        # draw bounding box for status
        if frame is not None and status_file is not None:
            status = status_data.get(frame_idx, last_status)
            last_status = status
            color = (0, 255, 0) if status else (255, 0, 0)
            h, w, _ = frame.shape
            thickness = 10
            cv2.rectangle(frame, (0, 0), (w, h), color, thickness, lineType=cv2.LINE_8)

        frames.append(frame)

    # create a moviepy clip from the frames
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_video, codec="libx264")
    print(f"Video saved as {output_video}")


def draw_arrow_on_frame(frame, yaw_rate, scale_factor=2):
    frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    h, w, _ = frame.shape
    center = (w // 2, h)
    length = h // 3  # arrow length

    # calculate arrow end point based on yaw rate
    end_point = (int(center[0] + length * np.sin(yaw_rate)), int(center[1] - length * np.cos(yaw_rate)))
    # draw the arrow
    frame_with_arrow = cv2.arrowedLine(
        frame.copy(),
        center,
        end_point,
        color=(255, 255, 255),
        thickness=length // 10,
        line_type=cv2.LINE_8,
        tipLength=0.3,
    )
    return frame_with_arrow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder", type=str)
    parser.add_argument("output_video", type=str)
    parser.add_argument("--status_file", type=str, default=None)
    parser.add_argument("--yaw_rate_file", type=str, default=None)
    parser.add_argument("--frame_range", type=int, nargs=2, default=None)
    args = parser.parse_args()
    # image_folder = "logs/images/learningflight_network_0/validate/disparity"
    # yaw_rate_file = "logs/images/learningflight_network_0/validate/yaw_rate_pred/data.txt"
    # output_video = "logs/videos/.mp4"
    scale_factor = 2
    fps = 50

    create_video_with_yaw_rate(
        args.image_folder, args.output_video, args.status_file, args.yaw_rate_file, args.frame_range, scale_factor, fps
    )
