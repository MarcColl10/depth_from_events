import os
import torch
import argparse


def load_state_dict(file_path):
    return torch.load(file_path, map_location="cpu", weights_only=True)


def check_state_dicts_equal(folder_path):
    state_dicts = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pt"):
            file_path = os.path.join(folder_path, file_name)
            state_dict = load_state_dict(file_path)
            state_dicts.append(state_dict)

    if len(state_dicts) < 2:
        print("Not enough .pt files to compare.")
        return

    base_state_dict = state_dicts[0]
    for i, state_dict in enumerate(state_dicts[1:], start=1):
        for key in base_state_dict.keys():
            if not torch.equal(base_state_dict[key], state_dict[key]):
                print(f"State dicts are not equal. Difference found in file: {os.listdir(folder_path)[i]}")
                return

    print("All state dicts are equal.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=str)
    args = parser.parse_args()

    check_state_dicts_equal(args.folder_path)
