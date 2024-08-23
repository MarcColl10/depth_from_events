from dotmap import DotMap
from lightning import LightningModule
import torch


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

        # set visualization
        self.visualizing = "visualizer" in self.trainer.callbacks

    def shared_step(self, batch, batch_idx, stage):
        # training: get optimizer because manual optimization
        if stage == "train":
            optimizer = self.optimizers()

        # unpack
        frames, eofs = batch.frames, batch.eofs

        # go over sequence
        log = DotMap()
        for x, eof in zip(frames, eofs):
            # forward network
            yhat = self.network(x)

            # go over loss functions
            for name, loss_fn in self.loss_functions.items():
                # forward
                loss_fn(x, yhat)

                # backward if enough passes
                if loss_fn.passes == loss_fn.accumulation_window:
                    loss = loss_fn.backward()

                    # training: backprop and optimize
                    # TODO: scheduler, gradient clipping
                    if stage == "train":
                        optimizer.zero_grad()
                        self.manual_backward(loss)
                        optimizer.step()

                        # detach network state
                        self.network.detach()

                    # reset loss and log
                    # loss per tbptt window per batch sample
                    # default batch size (seq_len) gives same value but rounding errors
                    for name, value in loss_fn.compute_and_reset().items():
                        self.log(f"{stage}/{name}", value, batch_size=1, on_epoch=True, prog_bar=True)

            # add to log if visualizing
            if self.visualizing:
                log.frame += [x]
                log.flow += [yhat]
                log.counter += 1

            # reset if end of sequence
            if any(eof):
                self.network.reset()
                for loss_fn in self.loss_functions.values():
                    loss_fn.reset()

        return log if self.visualizing else None

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = self.optimizer(self.network.parameters())
        return optimizer
