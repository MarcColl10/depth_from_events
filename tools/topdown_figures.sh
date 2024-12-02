#!/usr/bin/env bash

input_root="/mnt/d/ssl_depth"
output_root="figures/topdown"
mkdir -p "${output_root}"

python tools/figure_topdown.py \
    "${input_root}/000_pretrainedlearning/rosbag2_2024-11-14-08-12-43_0.h5" \
    "${input_root}/vlc-record-2024-11-14-11h34m45s-rtsp___192.168.209.102_live1s1.sdp-.mp4" \
    "${input_root}/000_pretrainedlearning/Take 2024-11-14 08.13.27 AM.csv" \
    "${output_root}/pretrainedlearning.pdf"

# python tools/figure_topdown.py \
#     "${input_root}/000_fromscratch/rosbag2_2024-11-14-14-15-14_0.h5" \
#     "${input_root}/vlc-record-2024-11-14-11h34m45s-rtsp___192.168.209.102_live1s1.sdp-.mp4" \
#     "${input_root}/000_fromscratch/Take 2024-11-14 08.13.27 AM_002.csv" \
#     "${output_root}/fromscratch.pdf"

# python tools/figure_topdown.py \
#     "${input_root}/000_realsense_avoid/rosbag2_2024-11-13-11-24-06_0.h5" \
#     "${input_root}/vlc-record-2024-11-14-11h34m45s-rtsp___192.168.209.102_live1s1.sdp-.mp4" \
#     "${input_root}/000_realsense_avoid/Take 2024-11-13 10.04.40 AM_001.csv" \
#     "${output_root}/realsense.pdf"

# python tools/figure_topdown.py \
#     "${input_root}/001_uzhfpvfinetuned_fixed/rosbag2_2024-11-13-21-01-01_0.h5" \
#     "${input_root}/vlc-record-2024-11-14-11h34m45s-rtsp___192.168.209.102_live1s1.sdp-.mp4" \
#     "${input_root}/001_uzhfpvfinetuned_fixed/Take 2024-11-13 09.00.50 PM.csv" \
#     "${output_root}/czfinetuned50eps.pdf"

# python tools/figure_topdown.py \
#     "${input_root}/002_pretrainedfixed/rosbag2_2024-11-13-21-25-55_0.h5" \
#     "${input_root}/vlc-record-2024-11-14-11h34m45s-rtsp___192.168.209.102_live1s1.sdp-.mp4" \
#     "${input_root}/002_pretrainedfixed/Take 2024-11-13 09.26.43 PM.csv" \
#     "${output_root}/pretrainedfixed.pdf"
