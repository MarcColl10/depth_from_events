from dotmap import DotMap  # TODO: move to tensordict
import hydra
from hydra.utils import instantiate
from rich.progress import Progress
import torch


"""
Some comments:
- See TODO's in file here and minimal_train.yaml
- When compiling transform, it's almost as fast as flow, but we get a small stutter every epoch, why? Not the case without compiling transform
- With compilation: 7s for 6400 network forwards including learning, so ~900 Hz on 4090, > 80% GPU utilization
- Without compilation: 24s, so ~266 Hz on 4090, 40% GPU utilization
- Even when using unwrapped version of disparity net (see commented code), still warning about CUDAGraphs
- On Orin: 40s for 6400 network forwards including learning, so ~160 Hz, varying GPU utilization, between 60-100%
"""


@hydra.main(version_base=None, config_path="config", config_name="minimal_train")
def main(config):
    # hardcode device and precision/dtype
    device = torch.device("cuda")
    dtype = torch.float32
    # torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False

    # dataset and dataloader
    dataset = instantiate(config.dataset, dtype=dtype)
    dataloader = instantiate(config.dataloader, dataset)

    # network, trace to get parameter shapes of lazy modules
    network = instantiate(config.network)
    network.to(device, dtype)
    # with torch.no_grad():
    #     _, memory = network(torch.zeros(1, 2, *dataset.sensor_size, device=device, dtype=dtype))
    #     memory = torch.zeros_like(memory)
    network.trace(torch.zeros(1, 2, *dataset.sensor_size, device=device, dtype=dtype), device=device)
    print(f"\n{network}\n\nwith {sum(p.numel() for p in network.parameters())} parameters\n")

    # compile network
    network = torch.compile(network, fullgraph=True, mode="reduce-overhead")

    # disparity + pose to flow transform
    transform = instantiate(config.transform)
    transform.to(device, dtype)

    # compile transform
    # init of grid necessary for compilation
    # TODO: eventually combine transform and network into one module
    transform.init_grid(1, *dataset.sensor_size, device, dtype)
    transform = torch.compile(transform, fullgraph=True, mode="reduce-overhead")

    # loss function, optimizer, grad clipping
    loss_function = instantiate(config.loss_function)
    optimizer = instantiate(config.optimizer, network.parameters())
    clip_grad = instantiate(config.trainer.clip_grad)

    # compile loss function
    # TODO: not working yet, DotMaps?
    # loss_function = torch.compile(loss_function, fullgraph=True, mode="reduce-overhead")

    # training loop
    with Progress() as progress:
        epoch_task = progress.add_task("[cyan]epoch: 0/0 loss: 0.000", total=config.trainer.epochs)
        iteration_task = progress.add_task("[cyan]iter: 0/0", total=len(dataloader))

        # loop over epochs of same recording
        for e in range(config.trainer.epochs):
            progress.reset(iteration_task)

            # loop over chunks of recording
            epoch_loss, epoch_passes = 0, 0
            for i, batch in enumerate(dataloader):

                # unpack and move to device
                frames, auxs, eofs = batch.frames, batch.auxs, batch.eofs
                frames = frames.to(device, dtype)
                auxs = DotMap(events=auxs.events.to(device, dtype), counts=auxs.counts.to(device))  # integer counts
                K_rect = batch.K_rect.to(device, dtype)
                inv_K_rect = batch.inv_K_rect.to(device, dtype)

                # loop over steps in chunk
                for j, (frame, eof) in enumerate(zip(frames, eofs)):
                    # get auxiliary: events and counts
                    aux = DotMap({k: v[j] for k, v in auxs.items()})

                    # forward network
                    # disparity net, so (disparity, pose)
                    yhat = network(frame)
                    # yhat, memory = network(frame, memory)
                    # flow = network(frame)

                    # transform to flow
                    flow = transform(yhat, K_rect, inv_K_rect)

                    # forward loss function
                    loss_function(frame, aux, flow)

                    # backward if enough passes
                    # detach network after optimizer step (tbptt)
                    if loss_function.passes == loss_function.accumulation_window:
                        loss = loss_function.backward()
                        optimizer.zero_grad()
                        loss.backward()
                        clip_grad(network.parameters())
                        optimizer.step()
                        network.detach()
                        # memory.detach_()
                        loss_val = loss_function.compute_and_reset().get("cmax", 0)
                        epoch_loss += loss_val
                        epoch_passes += 1

                    # reset if end of sequence
                    if any(eof):
                        # network.reset()  # NOTE: resetting network gives slower compile
                        loss_function.reset()

                progress.update(iteration_task, description=f"[cyan]iter: {i + 1}/{len(dataloader)}", advance=1)
            progress.update(
                epoch_task,
                description=f"[cyan]epoch: {e + 1}/{config.trainer.epochs} loss: {epoch_loss / epoch_passes:.3f}",
                advance=1,
            )


if __name__ == "__main__":
    main()
