from copy import deepcopy
from pathlib import Path

import hydra
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf, open_dict
import wandb


@hydra.main(version_base=None, config_path="config", config_name="eval")  # only hydra config and overrides
def main(overrides):
    # get checkpoint
    api = wandb.Api()
    project_path = f"{overrides.wandb.entity}/{overrides.wandb.project}"
    checkpoint_path = f"{project_path}/model-{overrides.runid}:{overrides.checkpoint}"
    checkpoint = Path(api.artifact(checkpoint_path).download()) / "model.ckpt"

    # get training config and merge with overrides
    run = api.run(f"{project_path}/{overrides.runid}")
    config = OmegaConf.create(deepcopy(run.config))
    with open_dict(config):
        config.merge_with(overrides)

    # dataset + dataloader = lightning datamodule
    datamodule = instantiate(config.datamodule)

    # network + loss functions = lightning module
    network = instantiate(config.network)
    loss_functions = instantiate(config.loss_functions)
    litmodule = get_class(config.litmodule._target_).load_from_checkpoint(
        checkpoint,
        network=network,
        loss_functions=loss_functions,
        optimizer=None,
    )

    # callbacks
    callbacks = instantiate(config.callbacks)
    callbacks.pop("checkpoint", None)

    # trainer and validate!
    trainer = instantiate(config.trainer, logger=False, callbacks=[cb for cb in callbacks.values()])
    trainer.validate(litmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
