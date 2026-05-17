from copy import deepcopy
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
import torch
import wandb

from depth_from_events.callbacks import PosePlotter


def _select(config, key, default=None):
    return OmegaConf.select(config, key, default=default)


def _load_torch_checkpoint(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, weights_only=True, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(loaded):
    """
    Supports both:
      1. Lightning checkpoints:
            {"state_dict": ...}
      2. raw PyTorch state_dict files:
            {"network.layer.weight": tensor, ...}
    """
    if isinstance(loaded, dict) and "state_dict" in loaded:
        return loaded["state_dict"]

    return loaded


def _clean_state_dict_keys(state_dict):
    """
    Fix keys from torch.compile / wrapped networks.
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod", "network")
        new_state_dict[new_key] = value

    return new_state_dict


def _apply_state_dict_maps(state_dict, config):
    state_dict_maps = _select(config, "state_dict_maps", default=None)

    if state_dict_maps is None:
        return state_dict

    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        for before, after in state_dict_maps.items():
            if before in new_key:
                new_key = new_key.replace(before, after)

        new_state_dict[new_key] = value

    return new_state_dict


def _load_config_from_wandb(overrides):
    api = wandb.Api()

    project_path = f"{overrides.wandb.entity}/{overrides.wandb.project}"
    run = api.run(f"{project_path}/{overrides.runid}")

    return OmegaConf.create(deepcopy(dict(run.config)))


def _load_config(overrides):
    """
    Normally loads the training config from W&B run config.

    Optional:
      local_config_path=/path/to/full/config.yaml

    The local config must be a complete saved Hydra config, not just
    config/train.yaml with defaults.
    """
    local_config_path = _select(overrides, "local_config_path", default=None)

    if local_config_path is not None:
        local_config_path = Path(local_config_path)

        if not local_config_path.exists():
            raise FileNotFoundError(f"Local config not found: {local_config_path}")

        return OmegaConf.load(local_config_path)

    return _load_config_from_wandb(overrides)


def _download_wandb_checkpoint_if_needed(overrides):
    """
    Only download W&B artifact when local_state_dict is NOT provided.
    """
    if overrides.local_state_dict is not None:
        return None

    api = wandb.Api()

    project_path = f"{overrides.wandb.entity}/{overrides.wandb.project}"
    checkpoint_path = f"{project_path}/model-{overrides.runid}:{overrides.checkpoint}"

    return Path(api.artifact(checkpoint_path).download()) / "model.ckpt"


@hydra.main(version_base=None, config_path="config", config_name="validate")
def main(overrides):
    # set to prevent warning
    torch.set_float32_matmul_precision("high")

    # load config first
    config = _load_config(overrides)

    # remove config entries requested by command line, e.g. deletes=[datamodule,loss_functions]
    for key in overrides.deletes:
        config.pop(key, None)

    # merge validate overrides into loaded training config
    with open_dict(config):
        config.merge_with(overrides)

    # get checkpoint
    checkpoint = _download_wandb_checkpoint_if_needed(overrides)

    # dataset + dataloader = lightning datamodule
    datamodule = instantiate(config.datamodule)

    # network + transform + loss functions = lightning module
    network = instantiate(config.network)
    transform = instantiate(config.transform)
    loss_functions = instantiate(config.loss_functions)

    litmodule = instantiate(
        config.litmodule,
        network,
        transform,
        loss_functions,
        optimizer=None,
        scheduler=None,
    )

    # load local or W&B checkpoint
    if overrides.local_state_dict is not None:
        loaded = _load_torch_checkpoint(overrides.local_state_dict)
    else:
        loaded = _load_torch_checkpoint(checkpoint)

    state_dict = _extract_state_dict(loaded)
    state_dict = _clean_state_dict_keys(state_dict)
    state_dict = _apply_state_dict_maps(state_dict, config)

    litmodule.load_state_dict(state_dict)
    litmodule.eval()
    litmodule.freeze()

    # callbacks
    callbacks = instantiate(config.callbacks)

    if callbacks is None:
        callbacks = {}

    callbacks = dict(callbacks)

    # Do not save new checkpoints during validation
    callbacks.pop("checkpoint", None)

    # Enable pose plotting
    if _select(config, "plot_pose", default=False):
        callbacks["pose_plotter"] = PosePlotter(
            output_dir=_select(config, "pose_plot_dir", default="pose_plots"),
            max_batches=_select(config, "pose_plot_max_batches", default=None),
            stage="validate",
        )

    # Remove None callbacks
    callback_list = [cb for cb in callbacks.values() if cb is not None]

    # trainer and validate
    trainer = instantiate(
        config.trainer,
        logger=False,
        callbacks=callback_list,
    )

    trainer.validate(litmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()