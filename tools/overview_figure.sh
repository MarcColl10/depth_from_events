#!/usr/bin/env bash

model="network_12000"
keys_with_constant="[validate/flow,validate/flow_raw,validate/disparity,validate/disparity_raw,validate/disparity_gt,validate/color_gt,validate/cmax_image_warped_events_t,validate/cmax_accumulated_events,validate/events,validate/events_raw]"
# python validate.py runid=auo4i8d9 local_state_dict=runs/Nov14_08-12-49_event-orin-drone/${model}.pt +callbacks=image_log name=${model} callbacks.visualizer.keys=${keys_with_constant} +trainer.limit_val_batches=8

# zip 760-769, so 8 val batches (above) is enough (100 per batch)
imgs=("00760" "00761" "00762" "00763" "00764" "00765" "00766" "00767" "00768" "00769")
keys=("validate/flow" "validate/disparity" "validate/cmax_image_warped_events_t" "validate/cmax_accumulated_events" "validate/events" "validate/events_raw")
if [ -f logs/images/overview_figure.zip ]; then
    rm logs/images/overview_figure.zip
fi
for img in "${imgs[@]}"; do
    for key in "${keys[@]}"; do
        if [ -f logs/images/${model}/${key}/${img}.png ]; then
            zip logs/images/overview_figure.zip logs/images/${model}/${key}/${img}.png
        elif [ -f logs/images/${model}/${key}/${img}.csv ]; then
            zip logs/images/overview_figure.zip logs/images/${model}/${key}/${img}.csv
        fi
    done
done
