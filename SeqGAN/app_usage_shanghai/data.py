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


if __name__ == '__main__':
    train_seqs, train_user_ids = load_sequences(config.TRAIN_FILE)
    test_seqs, test_user_ids = load_sequences(config.TEST_FILE)
    print('train:', train_seqs.shape, 'test:', test_seqs.shape)
    print('region id range:', train_seqs.min(), train_seqs.max())
