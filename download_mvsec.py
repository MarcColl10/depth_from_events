from depth_from_events.mvsec import MvsecDataModule


if __name__ == "__main__":
    datamodule = MvsecDataModule(
        root_dir="data/raw/mvsec",
        time_window=0.01,  # s
        train_seq_len=100,
        train_crop=None,
        val_recordings=None,
        val_time=None,
        val_crop=None,
        rectify=True,
        augmentations=["backward", "polarity", "horizontal"],
        return_events=True,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        download=True,
    )
    datamodule.prepare_data()
