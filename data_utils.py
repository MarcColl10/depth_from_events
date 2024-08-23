from itertools import islice
import random

from dotmap import DotMap
import torch
from torch.utils.data import BatchSampler


def batched(iterable, n, drop_last=False):
    """
    https://docs.python.org/3/library/itertools.html#itertools.batched
    """

    iterator = iter(iterable)
    batches = []
    while batch := tuple(islice(iterator, n)):
        if len(batch) == n or not drop_last:
            batches.append(batch)
    return batches


class ConcatBatchSampler(BatchSampler):
    """
    Batch sampler over a ConcatDataset.
    Shuffles, drops incomplete batches and cuts off longer sequences in a batch.
    """

    def __init__(self, concat_dataset, batch_size, shuffle=False):
        self.concat_dataset = concat_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_lengths = [len(dataset) for dataset in self.concat_dataset.datasets]
        self.idx_mapping = [
            list(range(cl - l, cl)) for l, cl in zip(self.sequence_lengths, self.concat_dataset.cumulative_sizes)
        ]
        self.reset()

    def reset(self):
        # shuffle
        random.shuffle(self.idx_mapping) if self.shuffle else None
        # batch
        self.batched_idx_mapping = batched(self.idx_mapping, self.batch_size, drop_last=True)
        # get length: zip so shortest
        self.length = sum(min(len(e) for e in batch) for batch in self.batched_idx_mapping)

    def __len__(self):
        return self.length

    def __iter__(self):
        # reset
        self.reset()

        # iterate over batches
        for batch in self.batched_idx_mapping:
            for idxs in zip(*batch):
                yield idxs


def time_first_collate(batch):
    collated_batch = DotMap()
    for key in batch[0]:
        if key in ["frames"]:
            collated_batch[key] = torch.stack([sample[key] for sample in batch], dim=1)
        elif key in ["K_rect", "inv_K_rect"]:
            collated_batch[key] = torch.stack([sample[key] for sample in batch])  # constant over time
        elif key in ["eofs"]:
            collated_batch[key] = list(zip(*[sample[key] for sample in batch]))
        else:
            collated_batch[key] = [sample[key] for sample in batch]
    return collated_batch


def only_add_batch_dim(batch):
    for key in batch:
        if key in ["frames"]:
            batch[key] = batch[key].unsqueeze(1)
        elif key in ["K_rect", "inv_K_rect"]:
            batch[key] = batch[key].unsqueeze(0)
        elif key in ["eofs"]:
            batch[key] = [[sample] for sample in batch[key]]
    return batch
