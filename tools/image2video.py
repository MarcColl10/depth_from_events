from pathlib import Path

import hydra
from moviepy.editor import clips_array, ImageSequenceClip


@hydra.main(config_path=".", config_name="image2video")
def main(config):
    # paths
    root_dir = Path(config.root_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # if no indices, use all
    if config.i0 is None or config.i1 is None:
        clips = []
        for name in config.names:
            clips.append(ImageSequenceClip(str(root_dir / name), fps=config.fps))
    else:
        raise NotImplementedError("Not implemented yet.")

    # create video
    if config.cols is not None:
        clips = [clips[i : i + config.cols] for i in range(0, len(clips), config.cols)]
    else:
        clips = [clips]
    clip = clips_array(clips).resize(config.resize)
    clip.write_videofile(
        str((output_dir / config.name).with_suffix(".mp4")), codec="libx264", fps=config.fps, bitrate=config.bitrate
    )


if __name__ == "__main__":
    main()
