from pathlib import Path

import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import OmegaConf
import torch
import wandb


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config):
    # set to prevent warning
    torch.set_float32_matmul_precision("high")

    # get checkpoint if given
    if config.runid is not None and config.checkpoint is not None:
        api = wandb.Api()
        project_path = f"{config.wandb.entity}/{config.wandb.project}"
        checkpoint_path = f"{project_path}/model-{config.runid}:{config.checkpoint}"
        checkpoint = Path(api.artifact(checkpoint_path).download()) / "model.ckpt"
    else:
        checkpoint = None

    # reproducibility
    if config.trainer.deterministic:
        seed_everything(42, workers=True)

    # dataset + dataloader = lightning datamodule
    datamodule = instantiate(config.datamodule)

    # network + transform + loss functions + optimizer = lightning module
    network = instantiate(config.network)
    transform = instantiate(config.transform)
    loss_functions = instantiate(config.loss_functions)
    optimizer = instantiate(config.optimizer)
    scheduler = instantiate(config.scheduler)
    litmodule = instantiate(config.litmodule, network, transform, loss_functions, optimizer, scheduler)

    # load state dict from checkpoint
    # lighting load_from_checkpoint is not transparent enough
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, weights_only=True, map_location="cpu")["state_dict"]
        if "state_dict_maps" in config:  # temporary
            new_state_dict = {}
            for key in state_dict:
                new_key = key
                for before, after in config.state_dict_maps.items():
                    if before in key:
                        new_key = new_key.replace(before, after)
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict
        litmodule.load_state_dict(state_dict)

    # callbacks
    callbacks = instantiate(config.callbacks)

    # logger
    logger = instantiate(config.logger)
    if logger is not None:
        logger.log_hyperparams(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
        enable_checkpointing = True
    else:
        logger = False
        enable_checkpointing = False
        callbacks.pop("checkpoint", None)

    # trainer and train!
    trainer = instantiate(
        config.trainer,
        logger=logger,
        callbacks=[cb for cb in callbacks.values()],
        enable_checkpointing=enable_checkpointing,
    )
    trainer.fit(litmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
