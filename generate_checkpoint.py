from copy import deepcopy
from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict
import torch
import wandb


@hydra.main(version_base=None, config_path="config", config_name="generate_checkpoint")
def main(overrides):
    # get checkpoint
    api = wandb.Api()
    project_path = f"{overrides.wandb.entity}/{overrides.wandb.project}"
    checkpoint_path = f"{project_path}/model-{overrides.runid}:{overrides.checkpoint}"
    checkpoint = Path(api.artifact(checkpoint_path).download()) / "model.ckpt"

    # get training config and save original
    run = api.run(f"{project_path}/{overrides.runid}")
    config = OmegaConf.create(deepcopy(run.config))
    save_dir = Path(overrides.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # with open(save_dir / "config_orig.yaml", "w") as f:
    #     OmegaConf.save(config=config, f=f)

    # merge with overrides
    for key in overrides.deletes:
        config.pop(key, None)
    with open_dict(config):
        config.merge_with(overrides)

    # get state dict
    if overrides.local_state_dict is not None:
        state_dict = torch.load(overrides.local_state_dict, weights_only=True, map_location="cpu")
        new_state_dict = {}
        for key in state_dict:
            new_state_dict[key.replace("_orig_mod", "network")] = state_dict[key]
        state_dict = new_state_dict
    else:
        state_dict = torch.load(checkpoint, weights_only=True, map_location="cpu")["state_dict"]
        if "state_dict_maps" in overrides:  # temporary
            new_state_dict = {}
            for key in state_dict:
                new_key = key
                for before, after in config.state_dict_maps.items():
                    if before in key:
                        new_key = new_key.replace(before, after)
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

    # save config and state dict
    save_dir = Path(overrides.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w") as f:
        OmegaConf.save(config=config, f=f)
    torch.save(state_dict, save_dir / "state_dict.pth")

    # TODO: manually delete unnecessary keys?


if __name__ == "__main__":
    main()
