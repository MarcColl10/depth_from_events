from dotmap import DotMap
from lightning import LightningModule
import torch

import callbacks


class Train(LightningModule):
    def __init__(self, network, loss_functions, optimizer):
        super().__init__()

        self.network = network
        self.loss_functions = loss_functions
        self.optimizer = optimizer
        self.automatic_optimization = False  # manual because tbptt

    def setup(self, stage):
        # trace lazy modules if training (always for litmodule Train?)
        if stage == "fit":
            x = torch.zeros(self.trainer.datamodule.train_frame_shape, device=self.device)
            self.network.trace(x)

        # wandb model watching
        if self.logger is not None:
            self.logger.watch(self.network, log="all", log_freq=self.trainer.log_every_n_steps * 100)

        # set visualization
        self.visualizing = any([isinstance(cb, callbacks.LiveVisualizer) for cb in self.trainer.callbacks])

    def shared_step(self, batch, batch_idx, stage):
        # training: get optimizer because manual optimization
        if stage == "train":
            optimizer = self.optimizers()

        # unpack
        frames, auxs, eofs, rec = batch.frames, batch.auxs, batch.eofs, batch.recording

        # go over sequence
        log = DotMap()
        for i, (x, eof) in enumerate(zip(frames, eofs)):
            # get auxiliary
            aux = DotMap({k: v[i] for k, v in auxs.items()})

            # forward network
            yhat = self.network(x)

            # go over loss functions
            for name, loss_fn in self.loss_functions[stage].items():
                # forward
                loss_fn(x, aux, yhat)

                # backward if enough passes
                if loss_fn.passes == loss_fn.accumulation_window:
                    loss = loss_fn.backward()

                    # training: backprop and optimize
                    # TODO: scheduler
                    # TODO: proper way to deal with no events?
                    if stage == "train" and loss is not None:
                        optimizer.zero_grad()
                        self.manual_backward(loss)
                        self.clip_gradients(optimizer, gradient_clip_val=self.gradient_clip_val)
                        optimizer.step()

                        # detach network state
                        self.network.detach()

                    # reset loss and log
                    # loss per tbptt window per batch sample
                    # default batch size (seq_len) gives same value but rounding errors
                    for name, value in loss_fn.compute_and_reset().items():
                        if stage == "train" and value:
                            self.log(f"{stage}/{name}", value, batch_size=1, on_epoch=True, prog_bar=True)
                        elif stage == "validate" and value:
                            self.log(f"{stage}/{name}/{rec}", value, batch_size=1)  # on_epoch true by default
                            self.log(f"{stage}/{name}/mean", value, batch_size=1, prog_bar=True)

            # add to log if visualizing
            if self.visualizing:
                log.frame += [x]
                log.flow += [yhat]
                log.counter += 1

            # reset if end of sequence
            if any(eof):
                self.network.reset()
                for loss_fn in self.loss_functions[stage].values():
                    loss_fn.reset()

        return log if self.visualizing else None

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "validate")

    def configure_optimizers(self):
        self.gradient_clip_val = self.optimizer.keywords.pop("gradient_clip_val", 0.0)
        optimizer = self.optimizer(self.network.parameters())
        return optimizer
