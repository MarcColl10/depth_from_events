#!/usr/bin/env bash

models=("network_0" "network_3000" "network_6000" "network_9000" "network_12000" "fromscratch12000")
keys_with_constant="[validate/flow,validate/flow_raw,validate/disparity,validate/disparity_raw,validate/disparity_gt,validate/color_gt,validate/cmax_image_warped_events_t,validate/cmax_accumulated_events,validate/events]"
keys_no_constant="[validate/flow,validate/flow_raw,validate/disparity,validate/disparity_raw,validate/cmax_image_warped_events_t]"
for model in "${models[@]}"; do
    if [ "$model" == "fromscratch12000" ]; then
        python validate.py runid=auo4i8d9 local_state_dict=runs/Nov14_14-15-20_event-orin-drone/network_12000.pt +callbacks=image_log name=${model} callbacks.visualizer.keys=${keys_with_constant}
    else
        python validate.py runid=auo4i8d9 local_state_dict=runs/Nov14_08-12-49_event-orin-drone/${model}.pt +callbacks=image_log name=${model} callbacks.visualizer.keys=${keys_no_constant}
    fi
done
# python validate.py runid=auo4i8d9 local_state_dict=runs/Nov13_21-01-07_event-orin-drone/network_0.pt +callbacks=image_log name=czfinetuned50eps callbacks.visualizer.keys=[validate/disparity,validate/disparity_gt,validate/cmax_image_warped_events_t,validate/cmax_accumulated_events,validate/events]

# zip images 2472, 3232, 4971, 8695
# all +1
if [ -f logs/images/onboardtrained_oncztest.zip ]; then
    rm logs/images/onboardtrained_oncztest.zip
fi
disparity_imgs=("02473" "03233" "04972" "08696")
disparity_raw_imgs=("02473" "03233" "04972" "08696")
disparity_gt_imgs=("02473" "03234" "04972" "08695")
color_gt_imgs=("02473" "03234" "04972" "08695")
events_imgs=("02473" "03233" "04972" "08696")
cmax_image_warped_events_t_imgs=("02473" "03233" "04972" "08696")
cmax_accumulated_events_imgs=("02473" "03233" "04972" "08696")
flow_imgs=("02473" "03233" "04972" "08696")
flow_raw_imgs=("02473" "03233" "04972" "08696")

for model in "${models[@]}"; do
    for img in "${disparity_imgs[@]}"; do
        zip logs/images/onboardtrained_oncztest.zip logs/images/${model}/validate/disparity/${img}.png
    done
    for img in "${disparity_raw_imgs[@]}"; do
        zip logs/images/onboardtrained_oncztest.zip logs/images/${model}/validate/disparity_raw/${img}.npy
    done
    for img in "${disparity_gt_imgs[@]}"; do
        zip logs/images/onboardtrained_oncztest.zip logs/images/${model}/validate/disparity_gt/${img}.png
    done
    for img in "${color_gt_imgs[@]}"; do
        zip logs/images/onboardtrained_oncztest.zip logs/images/${model}/validate/color_gt/${img}.png
    done
    for img in "${events_imgs[@]}"; do
        zip logs/images/onboardtrained_oncztest.zip logs/images/${model}/validate/events/${img}.png
    done
    for img in "${cmax_image_warped_events_t_imgs[@]}"; do
        zip logs/images/onboardtrained_oncztest.zip logs/images/${model}/validate/cmax_image_warped_events_t/${img}.png
    done
    for img in "${cmax_accumulated_events_imgs[@]}"; do
        zip logs/images/onboardtrained_oncztest.zip logs/images/${model}/validate/cmax_accumulated_events/${img}.png
    done
    for img in "${flow_imgs[@]}"; do
        zip logs/images/onboardtrained_oncztest.zip logs/images/${model}/validate/flow/${img}.png
    done
    for img in "${flow_raw_imgs[@]}"; do
        zip logs/images/onboardtrained_oncztest.zip logs/images/${model}/validate/flow_raw/${img}.npy
    done
done

# zip logs/images/onboardtrained_oncztest.zip logs/images/network0/validate/disparity/02473.png logs/images/network0/validate/disparity/03233.png logs/images/network0/validate/disparity/04972.png logs/images/network0/validate/disparity/08696.png
# zip logs/images/onboardtrained_oncztest.zip logs/images/network0/validate/disparity_gt/02473.png logs/images/network0/validate/disparity_gt/03234.png logs/images/network0/validate/disparity_gt/04972.png logs/images/network0/validate/disparity_gt/08695.png
# zip logs/images/onboardtrained_oncztest.zip logs/images/network0/validate/color_gt/02473.png logs/images/network0/validate/color_gt/03234.png logs/images/network0/validate/color_gt/04972.png logs/images/network0/validate/color_gt/08695.png
# zip logs/images/onboardtrained_oncztest.zip logs/images/network3000/validate/disparity/02473.png logs/images/network3000/validate/disparity/03233.png logs/images/network3000/validate/disparity/04972.png logs/images/network3000/validate/disparity/08696.png
# zip logs/images/onboardtrained_oncztest.zip logs/images/network6000/validate/disparity/02473.png logs/images/network6000/validate/disparity/03233.png logs/images/network6000/validate/disparity/04972.png logs/images/network6000/validate/disparity/08696.png
# zip logs/images/onboardtrained_oncztest.zip logs/images/network9000/validate/disparity/02473.png logs/images/network9000/validate/disparity/03233.png logs/images/network9000/validate/disparity/04972.png logs/images/network9000/validate/disparity/08696.png
# zip logs/images/onboardtrained_oncztest.zip logs/images/network12000/validate/disparity/02473.png logs/images/network12000/validate/disparity/03233.png logs/images/network12000/validate/disparity/04972.png logs/images/network12000/validate/disparity/08696.png
# zip logs/images/onboardtrained_oncztest.zip logs/images/fromscratch12000/validate/disparity/02473.png logs/images/fromscratch12000/validate/disparity/03233.png logs/images/fromscratch12000/validate/disparity/04972.png logs/images/fromscratch12000/validate/disparity/08696.png
# zip logs/images/onboardtrained_oncztest.zip logs/images/czfinetuned50eps/validate/disparity/02473.png logs/images/czfinetuned50eps/validate/disparity/03233.png logs/images/czfinetuned50eps/validate/disparity/04972.png logs/images/czfinetuned50eps/validate/disparity/08696.png
