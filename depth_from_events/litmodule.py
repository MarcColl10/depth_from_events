from collections import OrderedDict

from dotmap import DotMap
from lightning import LightningModule
import torch

from . import callbacks


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

        # NOTE: not helping!
        # # compile network
        # self.network = torch.compile(self.network, fullgraph=True, mode="reduce-overhead")

        # # compile transform
        # b, _, h, w = self.trainer.datamodule.train_frame_shape
        # self.transform.init_grid(b, h, w, self.device, torch.float32)
        # self.transform = torch.compile(self.transform, fullgraph=True, mode="reduce-overhead")

        # wandb model watching
        if self.logger is not None:
            self.logger.watch(self.network, log="all", log_freq=self.trainer.log_every_n_steps * 100)

        # set visualization
        self.visualizing = any(
            [isinstance(cb, (callbacks.LiveVisualizer, callbacks.ImageLogger)) for cb in self.trainer.callbacks]
        )

    def shared_step(self, batch, batch_idx, stage):
        # training: get optimizer because manual optimization
        if stage == "train":
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers() if self.scheduler is not None else None

        # unpack
        frames, auxs, eofs, rec = batch.frames, batch.auxs, batch.eofs, batch.recording

        # if has gt pose
        if "pose" in batch:
            pose_gt = batch.pose
        else:
            pose_gt = None

        # if has targets
        if "targets" in batch:
            targets = batch.targets
        else:
            targets = None

        # go over sequence
        log_seq = OrderedDict()
        for i, (frame, eof) in enumerate(zip(frames, eofs)):
            log_seq[i] = DotMap()
            log = log_seq[i]
            # get auxiliary: events and counts
            aux = DotMap({k: v[i] for k, v in auxs.items()})

            # forward network
            # if flow net, this is flow; else (depth/disparity, pose)
            yhat = self.network(frame)

            # transform network output
            if self.transform is not None:
                if len(yhat) == 2:
                    depth, pose = yhat
                elif len(yhat) == 3:
                    depth, pose, _ = yhat

                # override pose estimation if desired
                # if self.override_pose and pose_gt is not None:
                #     pose = pose_gt[i]
                #     yhat_list = list(yhat)
                #     yhat_list[1] = pose
                #     yhat = tuple(yhat_list)
                if "gt_rotation" in aux:
                    _, translation = pose.split([3, 3], dim=-1)
                    # rmat = self.transform.rodrigues(angle)
                    # angle = R.from_matrix(rmat.cpu().numpy()).as_euler("xyz", degrees=True)
                    # angle = R.from_rotvec(axisangle.cpu().numpy()).as_euler("xyz", degrees=True)
                    # gt_angle = R.from_rotvec(aux.gt_rotation.cpu().numpy()).as_euler("xyz", degrees=True)
                    # gt_angle = aux.gt_rotation
                    # gt_rmat = self.transform.rodrigues(gt_angle)
                    # gt_angle = R.from_matrix(gt_rmat.cpu().numpy()).as_euler("xyz", degrees=True)
                    pose = torch.cat([aux.gt_rotation, translation], dim=-1)
                    yhat = tuple([yhat[0], pose])
                    # log["/rotation_x"] = angle[:, 0].item()
                    # log["/rotation_y"] = angle[:, 1].item()
                    # log["/rotation_z"] = angle[:, 2].item()
                    # log["/rotation_gt_x"] = gt_angle[:, 0].item()
                    # log["/rotation_gt_y"] = gt_angle[:, 1].item()
                    # log["/rotation_gt_z"] = gt_angle[:, 2].item()

                flow = self.transform(yhat, batch.K_rect, batch.inv_K_rect)
                if self.network.mode == "depth":
                    # depth = self.transform.clip_depth(depth)
                    disparity = self.transform.depth_to_disparity(depth)  # TODO: also scaling?
                elif self.network.mode == "disparity":
                    disparity, depth = self.transform.disparity_to_depth(depth)
                self.log(f"{stage}/depth_std", depth.std(), batch_size=1, prog_bar=True)
            else:
                depth, disparity, pose = None, None, None
                flow = yhat

            # log model prediction
            self.log(f"{stage}/flow_abs_mean", flow.abs().mean(), batch_size=1, prog_bar=True)

            # add to log if visualizing
            if self.visualizing:
                log[f"{stage}/events"] = frame
                log[f"{stage}/flow"] = flow
                if self.transform is not None:
                    log[f"{stage}/disparity"] = disparity
                    log[f"{stage}/pose"] = pose
                if pose_gt is not None:
                    log["/pose_gt"] = pose_gt[i].unsqueeze(0)
                if targets is not None:
                    if targets[i].get("gt_depth") is not None:
                        log["/disparity_gt"] = self.transform.depth_to_disparity(targets[i].gt_depth)
                    elif targets[i].get("gt_disparity") is not None:
                        log["/disparity_gt"] = targets[i].gt_disparity

            # go over loss functions
            loss = 0
            for name, loss_fn in self.loss_functions[stage].items():
                # forward
                if name in ["cmax", "rsat"]:
                    loss_fn(frame, aux, flow)
                elif name in ["ea_smooth"]:
                    loss_fn(frame, disparity)
                elif name in ["scale_consistency"]:
                    loss_fn(disparity, pose, batch.K_rect)
                elif targets and name in ["depth_disparity"]:
                    if targets[i].get("gt_depth") is not None:
                        loss_fn(frame, depth, targets[i].gt_depth)
                    elif targets[i].get("gt_disparity") is not None:
                        loss_fn(frame, disparity, targets[i].gt_disparity)
                    elif targets[i].get("eval_disparity_id") is not None:
                        loss_fn(frame, disparity, targets[i].eval_disparity_id)

                # add to log if visualizing
                if self.visualizing:
                    if name in ["cmax", "rsat"]:
                        with torch.no_grad():
                            log[f"{stage}/{name}_accumulated_events"] = loss_fn.get_accumulated_events()
                            log[f"{stage}/{name}_image_warped_events_0"] = loss_fn.compute_iwe(0)
                            log[f"{stage}/{name}_image_warped_events_t"] = loss_fn.compute_iwe(loss_fn.passes)

                # backward if enough passes
                if loss_fn.passes == loss_fn.accumulation_window:
                    dloss = loss_fn.backward()
                    loss += dloss if dloss is not None else 0

            # training: backprop and optimize
            if stage == "train" and loss:
                optimizer.zero_grad()
                self.manual_backward(loss)
                self.clip_gradients(optimizer, gradient_clip_val=self.gradient_clip_val)
                optimizer.step()
                self.log("train/lr", scheduler.get_last_lr()[0]) if scheduler is not None else None
                scheduler.step() if scheduler is not None else None

                # detach network state
                self.network.detach()

            # go over loss functions
            for name, loss_fn in self.loss_functions[stage].items():
                # reset if enough passes
                if loss_fn.passes == loss_fn.accumulation_window:
                    # reset loss and log
                    # loss per tbptt window per batch sample
                    # default batch size (seq_len) gives same value but rounding errors
                    for name, value in loss_fn.compute_and_reset().items():
                        if stage == "train" and value:
                            self.log(f"{stage}/{name}", value, batch_size=1, on_epoch=True, prog_bar=True)
                        elif stage == "validate" and value:
                            self.log(f"{stage}/{name}/{rec}", value, batch_size=1)  # on_epoch true by default
                            self.log(f"{stage}/{name}/mean", value, batch_size=1)
                        elif stage == "test" and value:
                            if name.startswith("depth_disparity") and isinstance(value, tuple):
                                log[name] = value
                            else:
                                self.log(f"{stage}/{name}/{rec}", value, batch_size=1)
                                self.log(f"{stage}/{name}/mean", value, batch_size=1, prog_bar=True)

            # reset if end of sequence
            if any(eof):
                self.network.reset()
                for loss_fn in self.loss_functions[stage].values():
                    loss_fn.reset()

        return log_seq if self.visualizing or stage == "test" else None

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "validate")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")

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
