"""Analyze Motion/beijing_trajectories.json (one predicted 24h trajectory per user) against
the real Beijing mobility test set, using the same metrics as MoveSim/SeqGAN's next-location
comparison: mean_error_m, ratio_to_range, top1_acc, disp/mean_disp, disp/max_disp.

Each user in the JSON has exactly one predicted 24-hour sequence. It corresponds to that
user's LAST (most recent) day in the test set, so each predicted sequence is scored against
only that one real day.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOVESIM_DIR = os.path.abspath(os.path.join(HERE, '..', 'MoveSim', 'beijing'))
sys.path.insert(0, MOVESIM_DIR)
import config
import data
import geo

MOTION_FILE = os.path.join(HERE, 'beijing_trajectories.json')


def load_last_day_per_user(path, seq_len=config.SEQ_LEN):
    """Same grouping as data.load_sequences, but keeps the date and, per user, keeps only
    the chronologically last complete day."""
    groups = defaultdict(dict)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['user_id'], row['date'])
            groups[key][int(row['time'])] = int(row['region_id'])

    user_to_dates = defaultdict(dict)
    for (uid, date), time_to_region in groups.items():
        if len(time_to_region) != seq_len or set(time_to_region.keys()) != set(range(seq_len)):
            continue
        user_to_dates[uid][date] = [time_to_region[t] for t in range(seq_len)]

    last_day = {}
    for uid, date_to_seq in user_to_dates.items():
        last_date = max(date_to_seq.keys())
        last_day[uid] = (last_date, np.array(date_to_seq[last_date], dtype=np.int64))
    return last_day


def main():
    coords = geo.load_region_coords()
    scale = geo.range_scale()

    test_seqs, test_user_ids = data.load_sequences(config.TEST_FILE)
    mean_map, max_map = geo.user_displacement_stats(coords, test_seqs, test_user_ids)

    with open(MOTION_FILE) as f:
        motion = json.load(f)

    last_day = load_last_day_per_user(config.TEST_FILE)

    missing = [uid for uid in motion if uid not in last_day]
    if missing:
        print('WARNING: %d/%d motion users have no real test day: %s' %
              (len(missing), len(motion), missing[:10]))

    pred_list, true_list, uid_list, date_list = [], [], [], []
    for uid_str, pred_seq in motion.items():
        if uid_str not in last_day:
            continue
        last_date, true_seq = last_day[uid_str]
        pred_list.append(np.array(pred_seq, dtype=np.int64))
        true_list.append(true_seq)
        uid_list.append(uid_str)
        date_list.append(last_date)

    pred = np.stack(pred_list)   # [N, 24]
    true = np.stack(true_list)   # [N, 24]
    uids = np.array(uid_list)

    def report(pred, true, uids, label, hour_range):
        errs = geo.region_distance_matrix_lookup(coords, pred, true)   # [N, len(hour_range)]
        correct = (pred == true)
        acc = correct.mean()
        mean_err = errs.mean()

        ratio_mean, ratio_max = [], []
        for row, uid in zip(errs, uids):
            m = mean_map.get(str(uid))
            mx = max_map.get(str(uid))
            if m:
                ratio_mean.extend((row / m).tolist())
            if mx:
                ratio_max.extend((row / mx).tolist())
        rmean = float(np.mean(ratio_mean)) if ratio_mean else float('nan')
        rmax = float(np.mean(ratio_max)) if ratio_max else float('nan')

        print('%-28s mean_error_m: %8.2f  ratio_to_range: %.4f  top1_acc: %.4f  '
              'disp/mean_disp: %.4f  disp/max_disp: %.4f  (n=%d hours=%s)' %
              (label, mean_err, mean_err / scale, acc, rmean, rmax, errs.size, hour_range))

        per_hour_acc = correct.mean(axis=0)
        print('  per-hour top1_acc:', ' '.join('%.2f' % a for a in per_hour_acc))
        return {'mean_error_m': mean_err, 'ratio_to_range': mean_err / scale, 'top1_acc': acc,
                'ratio_mean_disp': rmean, 'ratio_max_disp': rmax}

    print('range_scale (sqrt(width*height), m): %.2f' % scale)
    print('motion users: %d, matched to real last-day sequences: %d' % (len(motion), pred.shape[0]))
    print()

    print('[All 24 hours, predicted trajectory vs. that user\'s last real test day]')
    report(pred, true, uids, 'Motion (last day, all hours)', '0..23')
    print()

    print('[Hours 1..23 only -- matches MoveSim/SeqGAN next-location-prediction protocol]')
    report(pred[:, 1:], true[:, 1:], uids, 'Motion (last day, hours 1..23)', '1..23')


if __name__ == '__main__':
    main()
