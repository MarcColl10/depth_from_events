import csv
from rich.progress import track
import numpy as np
import pandas as pd
from io import StringIO
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


def compute_cost(relative_rotation_quat, imu1_accel, imu2_accel):
    """
    Computes the cost as the discrepancy between accelerometer readings of two IMUs.

    Parameters:
        relative_rotation_quat (np.ndarray): Quaternion representing the relative rotation.
        imu1_accel (np.ndarray): Accelerometer readings from IMU 1 (N x 3 array).
        imu2_accel (np.ndarray): Accelerometer readings from IMU 2 (N x 3 array).

    Returns:
        float: The computed cost based on the accelerometer discrepancies.
    """
    # Convert quaternion to rotation matrix
    relative_rotation_quat /= np.linalg.norm(relative_rotation_quat)
    relative_rotation = R.from_quat(relative_rotation_quat).as_matrix()

    # Transform imu1 accelerations to imu2's frame
    transformed_imu1_accel = imu1_accel @ relative_rotation.T

    # Compute cost as the sum of squared differences in accelerations
    cost = np.sum((transformed_imu1_accel - imu2_accel) ** 2)
    return cost


def read_txt_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Remove the '#' character from the header line
    lines[0] = lines[0].replace("#", "").strip() + "\n"

    # Use StringIO to load the cleaned data into a pandas DataFrame
    cleaned_data = StringIO("".join(lines))
    df = pd.read_csv(cleaned_data, delimiter=" ")

    return df


def get_aligned_imu_data(sequence):
    davis_imu_data = read_txt_file(f"data/raw/uzh_fpv/indoor_forward/{sequence}_davis_with_gt/imu.txt")
    sd_imu_data = read_txt_file(f"data/raw/uzh_fpv/indoor_forward/{sequence}_snapdragon_with_gt/imu.txt")
    davis_gt = read_txt_file(f"data/raw/uzh_fpv/indoor_forward/{sequence}_davis_with_gt/groundtruth.txt")
    sd_gt = read_txt_file(f"data/raw/uzh_fpv/indoor_forward/{sequence}_snapdragon_with_gt/groundtruth.txt")

    # assert len(davis_gt) == len(sd_gt)
    assert abs(len(davis_gt) - len(sd_gt)) <= 1
    time_offset = davis_gt.iloc[0]["timestamp"] - sd_gt.iloc[0]["timestamp"]
    sd_imu_data["timestamp"] += time_offset

    N = 1000
    # randomly select N samples from davis_imu
    davis_imu_samples = davis_imu_data.sample(n=N)

    # Interpolate sd_imu_data at the timestamps of the davis_imu_samples
    sd_imu_interpolated = pd.DataFrame()
    for col in ["ang_vel_x", "ang_vel_y", "ang_vel_z", "lin_acc_x", "lin_acc_y", "lin_acc_z"]:
        sd_imu_interpolated[col] = np.interp(davis_imu_samples["timestamp"], sd_imu_data["timestamp"], sd_imu_data[col])

    davis_omega = davis_imu_samples[["ang_vel_x", "ang_vel_y", "ang_vel_z"]].to_numpy()
    sd_omega = sd_imu_interpolated[["ang_vel_x", "ang_vel_y", "ang_vel_z"]].to_numpy()

    davis_acc = davis_imu_samples[["lin_acc_x", "lin_acc_y", "lin_acc_z"]].to_numpy()
    sd_acc = sd_imu_interpolated[["lin_acc_x", "lin_acc_y", "lin_acc_z"]].to_numpy()

    # Combine the interpolated data with the davis_imu_samples
    # aligned_imu_data = pd.concat([davis_imu_samples.reset_index(drop=True), sd_imu_interpolated], axis=1)

    return davis_omega, sd_omega, davis_acc, sd_acc


def estimate_rotation(imu_1, imu_2):
    # Initial guess for relative rotation quaternion (identity rotation)
    initial_relative_rotation_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # Normalize quaternion to ensure valid rotation representation
    initial_relative_rotation_quat /= np.linalg.norm(initial_relative_rotation_quat)

    # Perform optimization to find the best-fit relative rotation quaternion
    result = minimize(
        compute_cost,
        initial_relative_rotation_quat,
        args=(imu_1, imu_2),
        method="BFGS",  # You can experiment with other methods like 'Nelder-Mead' or 'L-BFGS-B'
        options={"disp": True},
    )

    # Extract optimized quaternion and normalize it
    optimal_relative_rotation_quat = result.x / np.linalg.norm(result.x)

    # print("Optimal Relative Rotation Quaternion:", optimal_relative_rotation_quat)

    # Convert optimized quaternion to a rotation matrix for further use
    optimal_relative_rotation_matrix = R.from_quat(optimal_relative_rotation_quat)
    # print("Optimal Relative Rotation Matrix:\n", optimal_relative_rotation_matrix)

    return optimal_relative_rotation_matrix


if __name__ == "__main__":
    # sequence = "indoor_forward_5"

    # davis_omega, sd_omega, _, _ = get_aligned_imu_data(sequence)

    for sequence in [
        "indoor_forward_3",
        "indoor_forward_5",
        "indoor_forward_6",
        "indoor_forward_7",
        "indoor_forward_9",
        "indoor_forward_10",
    ]:
        davis_omega, sd_omega, _, _ = get_aligned_imu_data(sequence)
        relative_rotation_sn_2_davis_imu = estimate_rotation(davis_omega, sd_omega)
        print(f"Relative Rotation Matrix for {sequence}:\n{relative_rotation_sn_2_davis_imu.as_matrix()}\n")

        davis_gt = read_txt_file(f"data/raw/uzh_fpv/indoor_forward/{sequence}_davis_with_gt/groundtruth.txt")

        sd_gt = read_txt_file(f"data/raw/uzh_fpv/indoor_forward/{sequence}_snapdragon_with_gt/groundtruth.txt")

        davis_imu_2_cam = R.from_matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

        with open(f"data/raw/uzh_fpv/indoor_forward/{sequence}_davis_with_gt/groundtruth_corrected.txt", "w") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerow(["#", "timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
            for (_, row_sd), (_, row_davis) in track(zip(sd_gt.iterrows(), davis_gt.iterrows()), total=len(sd_gt)):
                world_sn = R.from_quat(row_sd[["qx", "qy", "qz", "qw"]].to_numpy())
                world_davis = world_sn * relative_rotation_sn_2_davis_imu
                q_davis = world_davis.as_quat()
                # print(f"q_davis as matrix: {world_davis.as_matrix()}")

                trans_davis = row_davis[["tx", "ty", "tz"]].to_numpy()
                time_davis = row_davis["timestamp"]

                # row_davis = np.concatenate((np.array([time_davis]), trans_davis, q_davis))

                formatted_row = [
                    f"{time_davis:.4f}",
                    *[f"{val:.10f}" for val in trans_davis],
                    *[f"{val:.10f}" for val in q_davis],
                ]
                writer.writerow(formatted_row)
                pass
