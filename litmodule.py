from dotmap import DotMap
from lightning import LightningModule
import torch

import callbacks


class Train(LightningModule):
    def __init__(self, network, transform, loss_functions, optimizer, scheduler):
        super().__init__()

        self.network = network
        self.transform = transform
        self.loss_functions = loss_functions
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.automatic_optimization = False  # manual because tbptt

    def setup(self, stage):
        # trace lazy modules if training (always for litmodule Train?)
        if stage == "fit":
            x = torch.zeros(self.trainer.datamodule.train_frame_shape, device=self.device)
            self.network.trace(x)

        # wandb model watching
        if self.logger is not None:
            self.logger.watch(self.network, log="all", log_freq=self.trainer.log_every_n_steps * 100)

        # convert to channels last
        # self.network.to(memory_format=torch.channels_last)

        # set visualization
        self.visualizing = any(
            [isinstance(cb, (callbacks.LiveVisualizer, callbacks.VideoLogger)) for cb in self.trainer.callbacks]
        )

    def shared_step(self, batch, batch_idx, stage):
        # training: get optimizer because manual optimization
        if stage == "train":
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers() if self.scheduler is not None else None

        # unpack
        frames, auxs, eofs, rec = batch.frames, batch.auxs, batch.eofs, batch.recording

        # go over sequence
        log = DotMap()
        for i, (frame, eof) in enumerate(zip(frames, eofs)):
            # get auxiliary: events and counts
            aux = DotMap({k: v[i] for k, v in auxs.items()})

            # forward network
            # if flow net, this is flow; else (disparity, pose)
            yhat = self.network(frame)

            # transform network output
            if self.transform is not None:
                if len(yhat) == 2:
                    disparity, pose = yhat
                elif len(yhat) == 3:
                    disparity, pose, _ = yhat
                flow = self.transform(yhat, batch.K_rect, batch.inv_K_rect)
            else:
                disparity, pose = None, None
                flow = yhat

            # log model prediction
            self.log(f"{stage}/flow_abs_mean", flow.abs().mean(), batch_size=1)

            # add to log if visualizing
            if self.visualizing:
                log.events += [frame]
                log.flow += [flow]
                if self.transform is not None:
                    log.disparity += [disparity]
                    log.pose += [pose]

            # go over loss functions
            for name, loss_fn in self.loss_functions[stage].items():
                # forward
                loss_fn(frame, aux, flow)

                # add to log if visualizing
                if self.visualizing:
                    with torch.no_grad():
                        log[f"{name}_accumulated_events"] += [loss_fn.get_accumulated_events()]
                        log[f"{name}_image_warped_events_0"] += [loss_fn.compute_iwe(0)]
                        log[f"{name}_image_warped_events_t"] += [loss_fn.compute_iwe(loss_fn.passes)]
                    # log[f"{name}_accumulated_flow_fw"] += [loss_fn.get_accumulated_flow(loss_fn.passes)]
                    # log[f"{name}_accumulated_flow_bw"] += [loss_fn.get_accumulated_flow(0)]

                # backward if enough passes
                if loss_fn.passes == loss_fn.accumulation_window:
                    loss = loss_fn.backward()

                    # training: backprop and optimize
                    if stage == "train" and loss is not None:
                        optimizer.zero_grad()
                        self.manual_backward(loss)
                        self.clip_gradients(optimizer, gradient_clip_val=self.gradient_clip_val)
                        optimizer.step()
                        self.log("train/lr", scheduler.get_last_lr()[0]) if scheduler is not None else None
                        scheduler.step() if scheduler is not None else None

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

                # else:
                #     self.network.detach()

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
        # split gradient clipping from optimizer
        self.gradient_clip_val = self.optimizer.keywords.pop("gradient_clip_val", 0.0)
        optimizer = self.optimizer(self.network.parameters())

        # scheduler: compute steps per epoch
        if self.scheduler is None:
            return optimizer
        else:
            dl_len = len(self.trainer.datamodule.train_dataloader())  # don't think this affects dl
            steps_per_seq = (
                self.trainer.datamodule.train_seq_len / self.loss_functions["train"]["cmax"].accumulation_window
            )
            steps_per_epoch = int(dl_len * steps_per_seq)
            scheduler = self.scheduler(optimizer, steps_per_epoch=steps_per_epoch)
            return [optimizer], [scheduler]
