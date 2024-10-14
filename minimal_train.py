from pathlib import Path

from dotmap import DotMap  # TODO: move to tensordict
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn
import torch
from torch.utils.tensorboard import SummaryWriter


"""
Some comments:
- See TODO's in file here and minimal_train.yaml
- When compiling transform, it's almost as fast as flow, but we get a small stutter every epoch, why? Not the case without compiling transform
- With compilation: 7s for 6400 network forwards including learning, so ~900 Hz on 4090, > 80% GPU utilization
- Without compilation: 24s, so ~266 Hz on 4090, 40% GPU utilization
- Even when using unwrapped version of disparity net (see commented code), still warning about CUDAGraphs
- On Orin: 40s for 6400 network forwards including learning, so ~160 Hz, varying GPU utilization, between 60-100%
"""


# from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten_dict(dictionary, parent_key=""):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + "." + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key).items())
        else:
            items.append((new_key, str(value)))
    return dict(items)


@hydra.main(version_base=None, config_path="config", config_name="minimal_train")
def main(config):
    # hardcode device and precision/dtype
    device = torch.device("cuda")
    dtype = torch.float32
    # torch.set_float32_matmul_precision("high")
    # torch.backends.cudnn.benchmark = False

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

    # tensorboard
    writer = SummaryWriter()
    writer.add_hparams(
        flatten_dict(OmegaConf.to_container(config, resolve=True)), {}, run_name=str(Path(writer.log_dir).absolute())
    )
    # writer.add_graph(network, torch.zeros(1, 2, *dataset.sensor_size, device=device, dtype=dtype))

    # save model state before training
    torch.save(network.state_dict(), f"{writer.log_dir}/network_0.pt")

    # training loop
    columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(
            style="bar.back", complete_style="bar.complete", finished_style="bar.finished", pulse_style="bar.pulse"
        ),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True),
    ]
    with Progress(*columns, speed_estimate_period=10) as progress:
        global_step_task = progress.add_task("[cyan]step: 0 loss: 0.000", total=config.trainer.n_steps)

        # train for certain amount of steps
        global_step = 0
        while True:

            # loop over chunks of recording
            for batch in dataloader:

                # unpack and move to device
                frames, auxs = batch.frames, batch.auxs
                frames = frames.to(device, dtype)
                auxs = DotMap(events=auxs.events.to(device, dtype), counts=auxs.counts.to(device))  # integer counts
                K_rect = batch.K_rect.to(device, dtype)
                inv_K_rect = batch.inv_K_rect.to(device, dtype)

                # loop over steps in chunk
                for j, frame in enumerate(frames):
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

                        # step logging
                        writer.add_scalar("loss_step", loss_val, global_step)
                        progress.update(
                            global_step_task,
                            description=f"[cyan]step: {global_step + loss_function.accumulation_window} loss: {loss_val:.3f}",
                            advance=loss_function.accumulation_window,
                        )
                        global_step += loss_function.accumulation_window
                        if global_step >= config.trainer.n_steps:
                            break

                    # pretend theres no end of sequence, no resetting
                    # if we drop incomplete batches this is fine

                # break loops
                else:
                    continue
                break
            else:
                continue
            break

    # save model
    torch.save(network.state_dict(), f"{writer.log_dir}/network_trained.pt")

    # close tensorboard
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
