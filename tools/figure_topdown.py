import argparse
from pathlib import Path

import cv2
import matplotlib
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import hdf5plugin

# set default font size
plt.rcParams.update({"font.size": 16})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5")
    parser.add_argument("mp4")
    parser.add_argument("csv")
    parser.add_argument("output")
    parser.add_argument("--t_offset", type=float, default=0.0)
    args = parser.parse_args()

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Show the image of cyberzoo top view and undistort it.
    # Load one frame of video .mp4 file
    cap = cv2.VideoCapture(args.mp4)
    ret, img = cap.read()
    cap.release()

    # store the image in the output folder
    cv2.imwrite(output_folder / "background.png", img)

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
    src_points = np.array(
        [[x1 - pad, y1 - pad], [x2 + pad, y2 - pad], [x3 + pad, y3 + pad], [x4 - pad, y4 + pad]], dtype=np.float32
    )

    pad = 80
    # Define the points in the "destination" perspective, a rectangle with straight lines
    dst_points = np.array(
        [
            [pad * wh_ratio, pad],
            [width - pad * wh_ratio, pad],
            [width - pad * wh_ratio, height - pad],
            [pad * wh_ratio, height - pad],
        ],
        dtype=np.float32,
    )

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
            scale = 1 + k * (r**2)
            new_x = center_x + dx * center_x * scale
            new_y = center_y + dy * center_y * scale

            # Set the remapped coordinates
            map_x[y, x] = new_x
            map_y[y, x] = new_y

    # Apply the remap to correct the barrel distortion
    corrected_image = cv2.remap(corrected_image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # Save the corrected image
    cv2.imwrite(output_folder / "background_corrected.png", corrected_image)

    # Determine where obstacle avoidance was active and mark in the plot.
    # Load rosbag with h5py
    f = h5py.File(args.h5)

    def time_to_seconds(time_str):
        m, s = map(int, time_str.split(":"))
        return m * 60 + s

    # Create dataframe with control values
    if "realsense" in args.output:
        control_values = f["control_rs"]
        control_df = pd.DataFrame(
            {"ts": control_values["yaw_rate_ts"][:] / 1000000, "status": np.ones_like(control_values["yaw_rate"][:])}
        )
        interventions = [
            ("00:00", "00:33"),
            ("01:02", "01:12"),
            ("01:46", "01:51"),
            ("02:08", "02:10"),
            ("02:25", "02:32"),
            ("03:03", "03:10"),
            ("03:29", "03:35"),
            ("03:52", "04:00"),
            ("04:42", "04:55"),
            ("04:58", "05:00"),
            ("05:27", "05:42"),
            ("05:55", "05:59"),
            ("06:41", "06:58"),
            ("07:35", "07:40"),
            ("08:13", "08:19"),
            ("08:54", "09:02"),
            # ("09:17", "09:30"),
            ("09:17", "10:00"),
        ]
        for start, end in interventions:
            start_idx = control_df[control_df["ts"] > time_to_seconds(start)].index[0]
            try:
                end_idx = control_df[control_df["ts"] > time_to_seconds(end)].index[0]
            except IndexError:
                end_idx = len(control_df)
            control_df.loc[start_idx:end_idx, "status"] = 0
    else:
        control_values = f["control"]
        control_df = pd.DataFrame(
            {"ts": control_values["status_ts"][:] / 1000000, "status": control_values["status"][:]}
        )

    # plt.figure()
    # plt.plot(f["control/status_ts"][:] / 1e6, f["control/status"][:], label="control status")
    # plt.plot(f["learner/yaw_rate_ts"][:] / 1e6, f["learner/yaw_rate"][:], label="yaw rate")
    # plt.plot(f["control_rs/yaw_rate_ts"][:] / 1e6, f["control_rs/yaw_rate"][:], label="gt yaw rate")
    # plt.legend()
    # plt.show()

    # # compute correlation between predicted and ground truth yaw rate when control is active
    # # control status 1 is active
    # control_active = control_df[control_df["status"] == 1]
    # yaw_rate = f["learner/yaw_rate"][:]
    # gt_yaw_rate = f["control_rs/yaw_rate"][:]
    # # print(np.corrcoef(yaw_rate[control_active.index], gt_yaw_rate[control_active.index]))

    # # Create a DataFrame with the yaw rates and timestamps
    # yaw_rate_df = pd.DataFrame({
    #     "ts": f["learner/yaw_rate_ts"][:] / 1e6,
    #     "yaw_rate": yaw_rate,
    #     "gt_yaw_rate": gt_yaw_rate
    # })

    # # Merge with control_active to get only the rows where control status is 1
    # merged_yaw_rate_df = pd.merge_asof(control_active, yaw_rate_df, on="ts", direction="nearest")

    # # Compute the correlation
    # correlation = merged_yaw_rate_df["learner_yaw_rate"].corr(merged_yaw_rate_df["control_rs_yaw_rate"])
    # print("Correlation between learner/yaw_rate and control_rs/yaw_rate when control/status is 1:", correlation)

    # Plot the X,Y coordinates of the drone on the image from Optitrack
    # Load the .csv file with optitrack data
    data = pd.read_csv(
        args.csv,
        delimiter=",",
        index_col=0,
        names=["ts", "Xr", "Yr", "Zr", "X", "Y", "Z"],
        skiprows=7,
    )

    # plot to find offset
    # use first altitude peak and compare with rosbag in foxglove
    plt.figure()
    plt.plot(data["ts"], data["Y"] / 100, label="optitrack z")
    plt.plot(control_df["ts"][:] + args.t_offset, control_df["status"][:], label="control status")
    plt.xlim(0, 300)
    plt.ylim(0, 10)
    # plt.show()
    # flush the plot
    plt.close()

    # offset to align the control data with the optitrack data
    control_df["ts"] = control_df["ts"] + args.t_offset

    # Merge control_df and data on the 'ts' column
    merged_df = pd.merge_asof(data, control_df, on="ts", direction="nearest")

    # start: column Y reaches value 200 for the first time
    # stop: column Y crosses value 200 for the second time
    start = ((merged_df["Y"].shift(100) < 100) & (merged_df["Y"] > 100)).idxmax()
    stop = ((merged_df["Y"].shift(100) > 100) & (merged_df["Y"] < 100)).idxmax()
    print(f"Start index: {start}")
    print(f"Stop index: {stop}")

    # relevant_df = merged_df[start:stop]
    # status_changes = relevant_df.index[relevant_df["status"].diff().fillna(0).ne(0)].tolist()

    plt.figure()
    plt.plot(merged_df["Y"])
    plt.plot(merged_df["status"] * 100)
    # for status_change in status_changes:
    #     plt.axvline(status_change, color="r")
    plt.axvline(start, color="r")
    plt.axvline(stop, color="r")
    # plt.show()
    # flush the plot
    plt.close()

    # Plot the merged data
    n_steps = stop - start
    plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB), alpha=0.5)

    # scale for image
    scale = 0.09  # scale factor to convert from meters to pixels
    x_offset = 163
    z_offset = 415
    merged_df["X"] = merged_df["X"] * scale
    merged_df["X"] = -merged_df["X"] + x_offset
    merged_df["Z"] = merged_df["Z"] * scale
    merged_df["Z"] = merged_df["Z"] + z_offset

    segments = np.zeros((len(range(start, stop, 10)), 2, 2))
    segments[:, 0, 0] = merged_df["Z"][start:stop:10]
    segments[:, 0, 1] = merged_df["X"][start:stop:10]
    segments[:, 1, 0] = merged_df["Z"][start + 10 : stop + 10 : 10]
    segments[:, 1, 1] = merged_df["X"][start + 10 : stop + 10 : 10]
    # segments = np.zeros((n_steps, 2, 2))
    # segments[:, 0, 0] = merged_df['Z'][:n_steps:]
    # segments[:, 0, 1] = merged_df['X'][:n_steps:]
    # segments[:, 1, 0] = merged_df['Z'][1:n_steps+1:]
    # segments[:, 1, 1] = merged_df['X'][1:n_steps+1:]
    cmap = matplotlib.colors.ListedColormap(["C1", "C0"])
    lc = LineCollection(segments, cmap=cmap, linewidth=2)
    lc.set_array(merged_df[start:stop:10]["status"])
    plt.gca().add_collection(lc)

    # Hide ticks and tick labels
    plt.gca().xaxis.set_ticks([])
    plt.gca().yaxis.set_ticks([])

    # Remove axis labels
    plt.gca().set_xlabel("")
    plt.gca().set_ylabel("")

    # labels
    labels = {
        "fromscratch": "TFS",
        "pretrainedfixed": "Only PT",
        "pretrainedlearning": "PT + OL",
        "realsense": "Using GT",
    }
    plt.title(labels[output_folder.stem])
    plt.savefig(output_folder / f"figure_{output_folder.stem}.pdf", bbox_inches="tight", transparent=True)

    # undo scaling
    merged_df["X"] = -merged_df["X"] + x_offset
    merged_df["X"] = merged_df["X"] / scale
    merged_df["Z"] = merged_df["Z"] - z_offset
    merged_df["Z"] = merged_df["Z"] / scale

    # convert xyz to meters
    merged_df["X"] = merged_df["X"] / 1000
    merged_df["Y"] = merged_df["Y"] / 1000
    merged_df["Z"] = merged_df["Z"] / 1000

    # write to csv file
    # write xyz and status columns to csv file
    # drop incomplete rows
    save_df = merged_df[start:stop].dropna()
    save_df.to_csv(output_folder / "3d_pos.csv", columns=["X", "Y", "Z", "status"], index=False)

    # compute distance between interventions
    relevant_df = merged_df[start:stop]
    status_changes = relevant_df.index[relevant_df["status"].diff().fillna(0).ne(0)].tolist()
    print(status_changes)
    total_distances = []

    # iterate through the status changes
    for i in range(0, len(status_changes) - 1, 2):
        start_idx = status_changes[i]
        end_idx = status_changes[i + 1]
        # print(f"Start index: {start_idx} going from {merged_df['status'].iloc[start_idx - 1]} to {merged_df['status'].iloc[start_idx]}")
        # print(f"End index: {end_idx} going from {merged_df['status'].iloc[end_idx - 1]} to {merged_df['status'].iloc[end_idx]}")

        # Ensure we are capturing the distance when status changes from 0 to 1 and back to 0
        if merged_df["status"].iloc[start_idx] == 1 and merged_df["status"].iloc[end_idx] == 0:
            # Calculate the distance covered in X and Z
            distance_x = merged_df["X"].iloc[start_idx:end_idx].diff().abs().sum()
            distance_z = merged_df["Z"].iloc[start_idx:end_idx].diff().abs().sum()
            distance = np.sqrt(distance_x**2 + distance_z**2)
            if distance > 1.05:
                total_distances.append(distance)

    print(f"Total distances: {sorted(total_distances)}")
    print(f"Median distance: {np.median(total_distances)}")
    print(f"Min distance: {np.min(total_distances)}")
    print(f"Mean distance: {np.mean(total_distances)}")
    print(f"Max distance: {np.max(total_distances)}")
    print(f"Sum of distances: {np.sum(total_distances)}")

    # boxplot of distances
    plt.figure()
    plt.boxplot(total_distances)
    plt.ylabel("Distance (m)")
    plt.ylim(0, 20)
    plt.savefig(output_folder / f"boxplot_{output_folder.stem}.pdf", bbox_inches="tight", transparent=True)

    # write to text file
    with open(output_folder / "distances.txt", "w") as f:
        f.writelines([f"{distance}\n" for distance in total_distances])

    print(args.output, merged_df[start:stop]["status"].mean())
