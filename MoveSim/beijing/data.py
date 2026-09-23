import csv
from collections import defaultdict

import numpy as np
import torch

import config


def load_sequences(path, seq_len=config.SEQ_LEN):
    """Group rows of a mobility csv by (user_id, date), sort by time, keep sequences
    of exactly seq_len entries. Returns (sequences, user_ids):
      sequences: int64 np.array [N, seq_len] of region ids.
      user_ids: np.array [N] of the user_id each sequence belongs to.
    """
    groups = defaultdict(dict)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['user_id'], row['date'])
            groups[key][int(row['time'])] = int(row['region_id'])

    sequences = []
    user_ids = []
    dropped = 0
    for key, time_to_region in groups.items():
        if len(time_to_region) != seq_len or set(time_to_region.keys()) != set(range(seq_len)):
            dropped += 1
            continue
        sequences.append([time_to_region[t] for t in range(seq_len)])
        user_ids.append(key[0])

    if dropped:
        print('load_sequences(%s): dropped %d incomplete groups, kept %d' % (path, dropped, len(sequences)))
    else:
        print('load_sequences(%s): kept %d complete sequences' % (path, len(sequences)))

    return np.array(sequences, dtype=np.int64), np.array(user_ids)


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = torch.from_numpy(sequences).long()

    def __len__(self):
        return self.sequences.size(0)

    def __getitem__(self, idx):
        return self.sequences[idx]


def make_loader(sequences, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False):
    ds = SequenceDataset(sequences)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def build_transition_matrix(sequences, num_regions=config.NUM_REGIONS):
    """M1: empirical bigram location-transition count matrix, built from every
    chronologically-consecutive pair of hours in `sequences`. Returns float32 [num_regions, num_regions].
    """
    prev = sequences[:, :-1].reshape(-1)
    nxt = sequences[:, 1:].reshape(-1)
    flat_idx = prev.astype(np.int64) * num_regions + nxt.astype(np.int64)
    counts = np.zeros(num_regions * num_regions, dtype=np.float32)
    np.add.at(counts, flat_idx, 1.0)
    return counts.reshape(num_regions, num_regions)


def start_location_distribution(sequences, num_regions=config.NUM_REGIONS):
    """Empirical histogram (normalized) of hour-0 regions, for the generator's free-running
    'real' starting-sample mode. Returns float32 [num_regions]."""
    counts = np.zeros(num_regions, dtype=np.float32)
    np.add.at(counts, sequences[:, 0], 1.0)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


if __name__ == '__main__':
    train_seqs, train_user_ids = load_sequences(config.TRAIN_FILE)
    test_seqs, test_user_ids = load_sequences(config.TEST_FILE)
    print('train:', train_seqs.shape, 'test:', test_seqs.shape)
    print('region id range:', train_seqs.min(), train_seqs.max())
    m1 = build_transition_matrix(train_seqs)
    print('M1 shape:', m1.shape, 'sum:', m1.sum(), 'expected:', train_seqs.shape[0] * (config.SEQ_LEN - 1))
