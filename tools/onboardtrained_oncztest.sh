#!/usr/bin/env bash

# python validate.py runid=auo4i8d9 local_state_dict=runs/Nov14_08-12-49_event-orin-drone/network_0.pt +callbacks=image_log name=network0 callbacks.visualizer.keys=[validate/disparity,validate/disparity_gt,validate/color_gt,validate/cmax_image_warped_events_t,validate/cmax_accumulated_events,validate/events]
# python validate.py runid=auo4i8d9 local_state_dict=runs/Nov14_08-12-49_event-orin-drone/network_3000.pt +callbacks=image_log name=network3000 callbacks.visualizer.keys=[validate/disparity,validate/disparity_gt,validate/cmax_image_warped_events_t,validate/cmax_accumulated_events,validate/events]
# python validate.py runid=auo4i8d9 local_state_dict=runs/Nov14_08-12-49_event-orin-drone/network_6000.pt +callbacks=image_log name=network6000 callbacks.visualizer.keys=[validate/disparity,validate/disparity_gt,validate/cmax_image_warped_events_t,validate/cmax_accumulated_events,validate/events]
# python validate.py runid=auo4i8d9 local_state_dict=runs/Nov14_08-12-49_event-orin-drone/network_9000.pt +callbacks=image_log name=network9000 callbacks.visualizer.keys=[validate/disparity,validate/disparity_gt,validate/cmax_image_warped_events_t,validate/cmax_accumulated_events,validate/events]
# python validate.py runid=auo4i8d9 local_state_dict=runs/Nov14_08-12-49_event-orin-drone/network_12000.pt +callbacks=image_log name=network12000 callbacks.visualizer.keys=[validate/disparity,validate/disparity_gt,validate/cmax_image_warped_events_t,validate/cmax_accumulated_events,validate/events]
# python validate.py runid=auo4i8d9 local_state_dict=runs/Nov14_14-15-20_event-orin-drone/network_12000.pt +callbacks=image_log name=fromscratch12000 callbacks.visualizer.keys=[validate/disparity,validate/disparity_gt,validate/cmax_image_warped_events_t,validate/cmax_accumulated_events,validate/events]
# python validate.py runid=auo4i8d9 local_state_dict=runs/Nov13_21-01-07_event-orin-drone/network_0.pt +callbacks=image_log name=czfinetuned50eps callbacks.visualizer.keys=[validate/disparity,validate/disparity_gt,validate/cmax_image_warped_events_t,validate/cmax_accumulated_events,validate/events]

# zip images 2472, 3232, 4971, 8695
# all +1
if [ -f logs/images/onboardtrained_oncztest.zip ]; then
    rm logs/images/onboardtrained_oncztest.zip
fi
zip logs/images/onboardtrained_oncztest.zip logs/images/network0/validate/disparity/02473.png logs/images/network0/validate/disparity/03233.png logs/images/network0/validate/disparity/04972.png logs/images/network0/validate/disparity/08696.png
zip logs/images/onboardtrained_oncztest.zip logs/images/network0/validate/disparity_gt/02473.png logs/images/network0/validate/disparity_gt/03234.png logs/images/network0/validate/disparity_gt/04972.png logs/images/network0/validate/disparity_gt/08695.png
zip logs/images/onboardtrained_oncztest.zip logs/images/network0/validate/color_gt/02473.png logs/images/network0/validate/color_gt/03234.png logs/images/network0/validate/color_gt/04972.png logs/images/network0/validate/color_gt/08695.png
zip logs/images/onboardtrained_oncztest.zip logs/images/network3000/validate/disparity/02473.png logs/images/network3000/validate/disparity/03233.png logs/images/network3000/validate/disparity/04972.png logs/images/network3000/validate/disparity/08696.png
zip logs/images/onboardtrained_oncztest.zip logs/images/network6000/validate/disparity/02473.png logs/images/network6000/validate/disparity/03233.png logs/images/network6000/validate/disparity/04972.png logs/images/network6000/validate/disparity/08696.png
zip logs/images/onboardtrained_oncztest.zip logs/images/network9000/validate/disparity/02473.png logs/images/network9000/validate/disparity/03233.png logs/images/network9000/validate/disparity/04972.png logs/images/network9000/validate/disparity/08696.png
zip logs/images/onboardtrained_oncztest.zip logs/images/network12000/validate/disparity/02473.png logs/images/network12000/validate/disparity/03233.png logs/images/network12000/validate/disparity/04972.png logs/images/network12000/validate/disparity/08696.png
zip logs/images/onboardtrained_oncztest.zip logs/images/fromscratch12000/validate/disparity/02473.png logs/images/fromscratch12000/validate/disparity/03233.png logs/images/fromscratch12000/validate/disparity/04972.png logs/images/fromscratch12000/validate/disparity/08696.png
zip logs/images/onboardtrained_oncztest.zip logs/images/czfinetuned50eps/validate/disparity/02473.png logs/images/czfinetuned50eps/validate/disparity/03233.png logs/images/czfinetuned50eps/validate/disparity/04972.png logs/images/czfinetuned50eps/validate/disparity/08696.png
