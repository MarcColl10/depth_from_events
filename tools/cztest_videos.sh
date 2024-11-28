#!/usr/bin/env bash

python tools/images_to_video.py \
    logs/images/fromscratch12000/validate/events \
    logs/videos/fromscratch12000_events.mp4

python tools/images_to_video.py \
    logs/images/fromscratch12000/validate/cmax_accumulated_events \
    logs/videos/fromscratch12000_cmax_accumulated_events.mp4

python tools/images_to_video.py \
    logs/images/fromscratch12000/validate/disparity_gt \
    logs/videos/fromscratch12000_disparity_gt.mp4

python tools/images_to_video.py \
    logs/images/fromscratch12000/validate/color_gt \
    logs/videos/fromscratch12000_color_gt.mp4

python tools/images_to_video.py \
    logs/images/network_12000/validate/disparity \
    logs/videos/network_12000_disparity.mp4

python tools/images_to_video.py \
    logs/images/network_12000/validate/cmax_image_warped_events_t \
    logs/videos/network_12000_cmax_image_warped_events_t.mp4
