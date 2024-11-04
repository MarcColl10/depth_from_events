from depth_from_events.dsec_old import DsecDataModule


if __name__ == "__main__":
    datamodule = DsecDataModule(
        root_dir="data/raw/dsec_old",
        time_window=10000,  # us
        count_thresh=100000,
        train_seq_len=100,
        train_crop=None,
        val_crop=None,
        rectify=True,
        augmentations=["backward", "polarity", "horizontal"],
        return_events=True,
        batch_size=4,
        shuffle=True,
        num_workers=8,
        download=True,
    )
    datamodule.prepare_data()
