import argparse
import re
from pathlib import Path

import cv2


def natural_key(path):
    nums = re.findall(r"\d+", path.name)
    return [int(n) for n in nums] if nums else [0]


def load_images_from_txt(seq_dir):
    txt = seq_dir / "images.txt"
    if not txt.exists():
        return []

    imgs = []
    for line in txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        png_tokens = [p for p in parts if p.lower().endswith((".png", ".jpg", ".jpeg"))]

        if png_tokens:
            p = seq_dir / png_tokens[-1]
        else:
            # Fallback for formats like: timestamp image_0_123.png
            p = seq_dir / parts[-1]

        if not p.exists():
            p = seq_dir / "img" / Path(p).name

        if p.exists():
            imgs.append(p)

    return imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq",
        default="indoor_forward_10_davis_with_gt",
        help="UZH-FPV sequence folder name",
    )
    parser.add_argument("--fps", type=float, default=30)
    args = parser.parse_args()

    seq_dir = Path("/data/marc/raw/uzh_fpv/indoor_forward") / args.seq
    out_path = Path("/data/marc/dataset_videos/uzh_fpv") / f"uzh_fpv_{args.seq}_frames.mp4"

    if not seq_dir.exists():
        raise FileNotFoundError(seq_dir)

    imgs = load_images_from_txt(seq_dir)

    if not imgs:
        imgs = sorted(
            list((seq_dir / "img").glob("*.png")) +
            list((seq_dir / "img").glob("*.jpg")) +
            list((seq_dir / "img").glob("*.jpeg")),
            key=natural_key,
        )

    print("Sequence:", seq_dir)
    print("Found images:", len(imgs))

    if not imgs:
        raise SystemExit("No image files found.")

    first = cv2.imread(str(imgs[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise RuntimeError(f"Could not read first image: {imgs[0]}")

    H, W = first.shape[:2]

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (W, H),
        True,
    )

    for i, p in enumerate(imgs):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        writer.write(img)

        if (i + 1) % 500 == 0 or i + 1 == len(imgs):
            print(f"{i + 1}/{len(imgs)}")

    writer.release()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
