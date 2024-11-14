from copy import deepcopy
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
import torch
import wandb


@hydra.main(version_base=None, config_path="config", config_name="validate")  # only hydra config and overrides
def main(overrides):
    # set to prevent warning
    torch.set_float32_matmul_precision("high")

    # get checkpoint
    api = wandb.Api()
    project_path = f"{overrides.wandb.entity}/{overrides.wandb.project}"
    checkpoint_path = f"{project_path}/model-{overrides.runid}:{overrides.checkpoint}"
    checkpoint = Path(api.artifact(checkpoint_path).download()) / "model.ckpt"

    # get training config and merge with overrides
    run = api.run(f"{project_path}/{overrides.runid}")
    config = OmegaConf.create(deepcopy(run.config))
    for key in overrides.deletes:
        config.pop(key, None)
    with open_dict(config):
        config.merge_with(overrides)

    # dataset + dataloader = lightning datamodule
    datamodule = instantiate(config.datamodule)

    # network + transform + loss functions = lightning module
    network = instantiate(config.network)
    transform = instantiate(config.transform)
    loss_functions = instantiate(config.loss_functions)
    litmodule = instantiate(config.litmodule, network, transform, loss_functions, optimizer=None, scheduler=None)
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
    litmodule.load_state_dict(state_dict)
    litmodule.eval()
    litmodule.freeze()

    # callbacks
    callbacks = instantiate(config.callbacks)
    callbacks.pop("checkpoint", None)

    # trainer and validate!
    trainer = instantiate(config.trainer, logger=False, callbacks=[cb for cb in callbacks.values()])
    trainer.validate(litmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
