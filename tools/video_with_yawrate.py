import os
import cv2
from moviepy.editor import VideoClip
import numpy as np
import pandas as pd


def create_video_with_yaw_rate(image_folder, yaw_rate_file, output_video, fps):
    # load yaw rate data
    yaw_data = pd.read_csv(yaw_rate_file, header=None, names=["frame", "yaw_rate"])
    yaw_data = yaw_data.set_index("frame").to_dict()["yaw_rate"]

    # get image files and sort them
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    image_indices = [int(os.path.splitext(f)[0]) for f in image_files]

    # read the first image to get dimensions
    first_image_path = os.path.join(image_folder, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # define the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # iterate over frames, handling missing ones
    last_frame = frame
    max_frame_index = max(image_indices)
    prev_yaw_rate = None
    for frame_index in range(max_frame_index + 1):
        if frame_index in image_indices:
            image_path = os.path.join(image_folder, f"{frame_index:05d}.png")
            frame = cv2.imread(image_path)
            last_frame = frame
        else:
            frame = last_frame

        # get yaw rate for this frame
        yaw_rate = yaw_data.get(frame_index, prev_yaw_rate)
        if yaw_rate is not None:
            # clamp to -0.5 to 0.5
            yaw_rate = np.clip(yaw_rate, -0.5, 0.5)
            prev_yaw_rate = yaw_rate
            # draw yaw rate as a vector on the frame
            center = (width // 2, height)
            length = 30  # length of the arrow
            angle = yaw_rate
            # angle = -yaw_rate * np.pi  # convert yaw rate to radians for drawing
            end_point = (int(center[0] + length * np.sin(angle)), int(center[1] - length * np.cos(angle)))
            cv2.arrowedLine(frame, center, end_point, (255, 255, 255), thickness=4, line_type=cv2.LINE_8, tipLength=0.5)

        # write frame to video
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as {output_video}")


# example usage
image_folder = "logs/images/learningflight_network_0/validate/disparity"
yaw_rate_file = "logs/images/learningflight_network_0/validate/yaw_rate_pred/data.txt"
output_video = "logs/videos/test.mp4"
fps = 50

create_video_with_yaw_rate(image_folder, yaw_rate_file, output_video, fps)
