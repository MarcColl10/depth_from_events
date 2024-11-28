#!/usr/bin/env bash

keys_with_constant="[validate/events,validate/disparity,validate/disparity_pred,validate/disparity_gt,validate/color_gt,validate/status,validate/yaw_rate,validate/yaw_rate_pred]"
models=("network_0" "network_12000")
for model in "${models[@]}"; do
    python validate.py \
        runid=auo4i8d9 \
        local_state_dict=runs/Nov14_08-12-49_event-orin-drone/${model}.pt \
        +datamodule.val_recordings="[[rosbag2_2024-11-14-08-12-43_0,[180e6,660e6]]]" \
        +callbacks=image_log \
        name=learningflight_${model} \
        callbacks.visualizer.keys=${keys_with_constant}
done

keys=("events" "disparity" "disparity_gt" "color_gt")
for key in "${keys[@]}"; do
    for model in "${models[@]}"; do
        if [ "$model" == "network_0" ]; then
            start_frame=0
            stop_frame=3000
        elif [ "$model" == "network_12000" ]; then
            start_frame=14870
            stop_frame=17870
        fi
        if [ "$key" == "disparity" ]; then
            python tools/images_to_video.py \
                logs/images/learningflight_${model}/validate/${key} \
                logs/videos/learningflight_${model}_${key}.mp4 \
                --status_file logs/images/learningflight_${model}/validate/status/data.txt \
                --yaw_rate_file logs/images/learningflight_${model}/validate/yaw_rate_pred/data.txt \
                --frame_range $start_frame $stop_frame
        elif [ "$key" == "disparity_gt" ]; then
            python tools/images_to_video.py \
                logs/images/learningflight_${model}/validate/${key} \
                logs/videos/learningflight_${model}_${key}.mp4 \
                --yaw_rate_file logs/images/learningflight_${model}/validate/yaw_rate/data.txt \
                --frame_range $start_frame $stop_frame
        else
            python tools/images_to_video.py \
                logs/images/learningflight_${model}/validate/${key} \
                logs/videos/learningflight_${model}_${key}.mp4 \
                --frame_range $start_frame $stop_frame
        fi
    done
done
