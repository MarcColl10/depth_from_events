# On-Device Self-Supervised Learning of Low-Latency Monocular Depth from Only Events

This repository contains the code for the paper "On-Device Self-Supervised Learning of Low-Latency Monocular Depth from Only Events", submitted to CVPR 2025. The code is structured as follows:
- [`config`](config): configuration files
- [`depth_from_events`](depth_from_events): main codebase
- [`checkpoints`](checkpoints): model checkpoints

To download MVSEC/DSEC, use [`download_mvsec.sh`](download_mvsec.sh) and [`download_dsec.sh`](download_dsec.sh) respectively.

The sequences recorded during the flight experiments can unfortunately not be shared due file size limitations and double blind policy. However, these will be made publicly available upon acceptance.

## Training a network

Configure training in [`configs/train.yaml`](config/train.yaml) and run:
```bash
python train.py
```

## Evaluating checkpoints

See the below commands for evaluating a specific checkpoint on a specific dataset. If you want visuals, add `+callbacks=live_vis` to the command.

Running checkpoint on UZH-FPV:
```bash
python validate.py runid=a2a4gwea +datamodule=uzh_fpv deletes=[datamodule,loss_functions] +state_dict_maps="{disp_decoder:depth_decoder}" +loss_functions@loss_functions.validate=[rsat]
```

Running UZH-FPV checkpoint on CZ flight data:
```bash
python validate.py runid=a2a4gwea +datamodule=flights deletes=[datamodule,loss_functions] +state_dict_maps="{disp_decoder:depth_decoder}" +loss_functions@loss_functions.validate=[rsat,depth_disparity]
```

Running first learning flight checkpoint on CZ flight data (same as UZH-FPV checkpoint above):
```bash
python validate.py runid=auo4i8d9 local_state_dict=runs/Nov14_08-12-49_event-orin-drone/network_0.pt +datamodule=flights deletes=[datamodule,loss_functions] +loss_functions@loss_functions.validate=[rsat,depth_disparity]
```

Running final learning flight checkpoint on CZ flight data:
```bash
python validate.py runid=auo4i8d9 local_state_dict=runs/Nov14_08-12-49_event-orin-drone/network_12000.pt +datamodule=flights deletes=[datamodule,loss_functions] +loss_functions@loss_functions.validate=[rsat,depth_disparity]
```

Running checkpoint on MVSEC:
```bash
python validate.py runid=jxg1ghsx +datamodule=mvsec deletes=[datamodule,loss_functions] +state_dict_maps="{disp_decoder:depth_decoder}" +loss_functions@loss_functions.validate=[rsat,depth_disparity] loss_functions.validate.depth_disparity.cut_offs=[10,20,30] loss_functions.validate.depth_disparity.mask_by_events=[true,false]
```

Running checkpoint on DSEC:
```bash
python validate.py runid=mwb18otp +datamodule=dsec deletes=[datamodule,loss_functions] +state_dict_maps="{disp_decoder:depth_decoder}" +loss_functions@loss_functions.validate=[rsat,depth_disparity]
```
