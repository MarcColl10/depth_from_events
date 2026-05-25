import h5py
import cv2
import numpy as np
from pathlib import Path

h5_path = Path("/data/marc/raw/mvsec/indoor_flying/indoor_flying1_data.hdf5")
out_path = Path("/data/marc/dataset_videos/mvsec_indoor_flying1_davis_left_cropped.mp4")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Same crop as mvsec.yaml: [top, left, bottom, right]
top, left, bottom, right = 0, 1, 192, 345

with h5py.File(h5_path, "r") as h:
    imgs = h["davis/left/image_raw"]
    ts = h["davis/left/image_raw_ts"]

    H = bottom - top
    W = right - left

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (W, H),
        True,
    )

    t0 = float(ts[0])

    for i in range(len(imgs)):
        img = imgs[i][top:bottom, left:right]

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # overlay real timestamp relative to start
        text = f"t = {float(ts[i]) - t0:.2f} s"
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        writer.write(img)

    writer.release()

print("Saved:", out_path)
