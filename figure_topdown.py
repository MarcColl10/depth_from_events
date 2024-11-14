import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import hdf5plugin


if __name__ == '__main__':

    # Show the image of cyberzoo top view and undistort it. 
    # Load one frame of video .mp4 file
    cap = cv2.VideoCapture('data/figures/top.mp4')
    ret, img = cap.read()
    cap.release()

    # Define width and height (based on measured size of the carpet)
    width = 790
    height = 594
    wh_ratio = width / height

    # Define four points in the image that form the distorted rectangle
    x1, y1 = 413, 189
    x2, y2 = 1488, 122
    x3, y3 = 1507, 895
    x4, y4 = 517, 969

    pad = 0
    # pad = 120
    src_points = np.float32([[x1 - pad, y1 - pad], [x2 + pad, y2 - pad], [x3 + pad, y3 + pad], [x4 - pad, y4 + pad]])

    pad = 80
    # Define the points in the "destination" perspective, a rectangle with straight lines
    dst_points = np.float32([
        [pad * wh_ratio, pad],
        [width - pad * wh_ratio, pad],
        [width - pad * wh_ratio, height - pad],
        [pad * wh_ratio, height - pad]
    ])

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation to correct the image
    corrected_image = cv2.warpPerspective(img, matrix, (width, height))

    # Define parameters for radial distortion correction
    center_x, center_y = width / 2, height / 2  # Assume the center of the image is the optical center
    k = -0.08  # Distortion coefficient (negative for barrel distortion)

    # Create remap coordinates
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            # Compute normalized coordinates with respect to the center
            dx = (x - center_x) / center_x
            dy = (y - center_y) / center_y
            r = np.sqrt(dx * dx + dy * dy)

            # Apply the radial distortion formula
            scale = 1 + k * (r ** 2)
            new_x = center_x + dx * center_x * scale
            new_y = center_y + dy * center_y * scale

            # Set the remapped coordinates
            map_x[y, x] = new_x
            map_y[y, x] = new_y

    # Apply the remap to correct the barrel distortion
    corrected_image = cv2.remap(corrected_image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # Plot the X,Y coordinates of the drone on the image from Optitrack
    # Load the .csv file with optitrack data
    data = pd.read_csv('data/figures/top_more.csv', delimiter=',', index_col=0, names=['ts', 'Xr', 'Yr', 'Zr', 'X', 'Y', 'Z'], skiprows=7)
    
    scale = 0.09 # scale factor to convert from meters to pixels
    x_offset = 163
    z_offset = 415
    data['X'] = data['X'] * scale
    data['X'] = -data['X'] + x_offset
    data['Z'] = data['Z'] * scale
    data['Z'] = data['Z'] + z_offset

    # Determine where obstacle avoidance was active and mark in the plot. 
    # Load rosbag with h5py    
    f = h5py.File('data/figures/top.h5', 'r')
    
    control_values = f["control"]

    # Create dataframe with control values
    control_df = pd.DataFrame({
        'ts': control_values['status_ts'][:] / 1000000,
        'status': control_values['status'][:]
    })

    t_offset = 14.8 # offset to align the control data with the optitrack data
    control_df['ts'] = control_df['ts'] - control_df['ts'][0] + t_offset

    # Merge control_df and data on the 'ts' column
    merged_df = pd.merge_asof(data, control_df, on='ts', direction='nearest')

    # Plot the merged data
    n_steps = 160000 # amount of steps to plot
    plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
    plt.plot(merged_df['Z'][:n_steps], merged_df['X'][:n_steps], 'r-', linewidth=4)
    plt.xlabel('Z')
    plt.ylabel('X')
    plt.plot(merged_df['Z'][0], merged_df['X'][0], 'bo')

    # Highlight points where obstacle avoidance was active
    active_points = merged_df[:n_steps][merged_df['status'][:n_steps] == 1]
    plt.plot(active_points['Z'], active_points['X'], 'go', markersize=2)

    plt.show()