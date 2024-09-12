import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config):
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

    # TODO: immediately validate/test/create videos


if __name__ == "__main__":
    main()
