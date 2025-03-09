import matplotlib.pyplot as plt
import numpy as np

from depth_from_events.dsec import DsecDataModule

"""
- we sample uniformly per recording, but repeat the recording n times, where n = seq duration / rec duration
- is this fair?
- only sampling chunks of 100 fits into memory, then split into 10x10
"""

if __name__ == "__main__":
    datamodule = DsecDataModule(
        root_dir="data/raw/dsec",
        time_window=10000,  # us
        count_thresh=None,
        train_seq_len=100,
        train_crop=None,
        val_crop=None,
        rectify=False,
        augmentations=None,
        return_events=True,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        download=False,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    samples = 10000
    n = 10
    sum = []
    minmax = []
    padding = []
    padding_frac = []

    for batch in loader:
        # 100 counts reshaped to 10x10
        counts = batch["auxs"]["counts"].squeeze().numpy()
        counts = counts.reshape(-1, n)
        sum.extend(counts.sum(1))
        # difference between max and min in each row
        minmax.extend(counts.max(1) - counts.min(1))
        # total padding in each row
        pad = (counts.max(1, keepdims=True) - counts).sum(1)
        padding.extend(pad)
        padding_frac.extend(pad / (counts.max(1) * n))
        samples -= 10
        print(f"{samples} samples left", end="\r")
        if samples <= 0:
            break

    sum = np.array(sum)
    minmax = np.array(minmax)
    padding = np.array(padding)
    padding_frac = np.array(padding_frac)
    print(f"sum: {sum.mean():.2f} ± {sum.std():.2f}")
    print(f"minmax: {minmax.mean():.2f} ± {minmax.std():.2f}")
    print(f"padding: {padding.mean():.2f} ± {padding.std():.2f}")
    print(f"padding_frac: {padding_frac.mean():.2f} ± {padding_frac.std():.2f}")
    print(f"for {len(minmax)} samples")

    fig, ax = plt.subplots(1, 4, figsize=(10, 3))
    ax[0].hist(sum, bins=30)
    ax[0].set_title("sum")
    ax[1].hist(minmax, bins=30)
    ax[1].set_title("minmax")
    ax[2].hist(padding, bins=30)
    ax[2].set_title("padding")
    ax[3].hist(padding_frac, bins=30)
    ax[3].set_title("padding_frac")
    fig.tight_layout()
    plt.savefig("dsec_padding.png", dpi=300)
