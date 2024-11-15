#!/usr/bin/env bash

python figure_topdown.py \
    "/mnt/d/000_pretrainedlearning/rosbag2_2024-11-14-08-12-43_0.h5" \
    "/mnt/d/vlc-record-2024-11-14-11h34m45s-rtsp___192.168.209.102_live1s1.sdp-.mp4" \
    "/mnt/d/000_pretrainedlearning/Take 2024-11-14 08.13.27 AM.csv" \
    "data/figures/pretrainedlearning.pdf"

# python figure_topdown.py "data/raw/flights/rosbag2_2024-11-14-14-15-14_0.h5" "data/figures/Take 2024-11-14 08.13.27 AM_002.csv" "data/figures/fromscratch.pdf"
python figure_topdown.py \
    "/mnt/d/000_fromscratch/rosbag2_2024-11-14-14-15-14_0.h5" \
    "/mnt/d/vlc-record-2024-11-14-11h34m45s-rtsp___192.168.209.102_live1s1.sdp-.mp4" \
    "/mnt/d/000_fromscratch/Take 2024-11-14 08.13.27 AM_002.csv" \
    "data/figures/fromscratch.pdf"

# python figure_topdown.py "data/raw/flights/rosbag2_2024-11-13-11-24-06_0.h5" "data/figures/Take 2024-11-13 10.04.40 AM_001.csv" "data/figures/realsense.pdf"
python figure_topdown.py \
    "/mnt/d/000_realsense_avoid/rosbag2_2024-11-13-11-24-06_0.h5" \
    "/mnt/d/vlc-record-2024-11-14-11h34m45s-rtsp___192.168.209.102_live1s1.sdp-.mp4" \
    "/mnt/d/000_realsense_avoid/Take 2024-11-13 10.04.40 AM_001.csv" \
    "data/figures/realsense.pdf"

# python figure_topdown.py "data/raw/flights/rosbag2_2024-11-13-21-01-01_0.h5" "data/figures/Take 2024-11-13 09.00.50 PM.csv" "data/figures/czfinetuned50eps.pdf"
python figure_topdown.py \
    "/mnt/d/001_uzhfpvfinetuned_fixed/rosbag2_2024-11-13-21-01-01_0.h5" \
    "/mnt/d/vlc-record-2024-11-14-11h34m45s-rtsp___192.168.209.102_live1s1.sdp-.mp4" \
    "/mnt/d/001_uzhfpvfinetuned_fixed/Take 2024-11-13 09.00.50 PM.csv" \
    "data/figures/czfinetuned50eps.pdf"

# python figure_topdown.py "data/raw/flights/rosbag2_2024-11-13-21-25-55_0.h5" "data/figures/Take 2024-11-13 09.26.43 PM.csv" "data/figures/pretrainedfixed.pdf"
python figure_topdown.py \
    "/mnt/d/002_pretrainedfixed/rosbag2_2024-11-13-21-25-55_0.h5" \
    "/mnt/d/vlc-record-2024-11-14-11h34m45s-rtsp___192.168.209.102_live1s1.sdp-.mp4" \
    "/mnt/d/002_pretrainedfixed/Take 2024-11-13 09.26.43 PM.csv" \
    "data/figures/pretrainedfixed.pdf"
