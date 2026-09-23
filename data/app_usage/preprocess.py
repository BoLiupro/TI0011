"""Convert raw per-second app_usage_records/*.csv (fine-grained base_id locations) into the
same hourly, gridded region-sequence format used by SeqGAN for the beijing/shanghai mobility
datasets: location.csv (region_id,lon_center,lat_center,<14 POI fraction cols>) and
mobility_train.csv/mobility_test.csv (user_id,date,time,region_id,region_poi).

Pipeline per city:
  1. Build a GRID_ROWS x GRID_COLS grid over the 1st-99th percentile lon/lat bounding box of
     location.csv (raw base stations have long-tail outliers far outside the real city area).
  2. Map every base_id -> region_id (drop base stations outside the bbox).
  3. Stream all app_usage_records/*.csv, map each event's location -> region_id, drop
     out-of-bbox events, derive (date, hour) from the time field.
  4. For each (user_id, date), take the majority region per observed hour, then impute missing
     hours by forward-fill then back-fill. Require >= MIN_OBSERVED_HOURS real (non-imputed)
     hours to keep a user-day.
  5. Split users into disjoint train/test sets.
  6. Write location.csv + mobility_train.csv + mobility_test.csv to data/app_usage/<city>/derived/.
"""
import argparse
import glob
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

GRID_ROWS = 20
GRID_COLS = 20
NUM_REGIONS = GRID_ROWS * GRID_COLS
SEQ_LEN = 24
MIN_OBSERVED_HOURS = 8
TEST_USER_FRAC = 0.15
SEED = 88

POI_COLS = [
    'Transportation Facilities', 'Leisure & Entertainment', 'Companies & Enterprises',
    'Healthcare', 'Commercial & Residential', 'Tourist Attractions', 'Automotive',
    'Life Services', 'Science & Education & Culture', 'Shopping & Consumer Goods',
    'Sports & Fitness', 'Hotels & Accommodations', 'Financial Institutions', 'Dining & Cuisine',
]


def build_grid(location_csv):
    loc = pd.read_csv(location_csv)
    lon_min, lon_max = loc['longitude'].quantile([0.01, 0.99])
    lat_min, lat_max = loc['latitude'].quantile([0.01, 0.99])

    in_bbox = (
        (loc['longitude'] >= lon_min) & (loc['longitude'] <= lon_max) &
        (loc['latitude'] >= lat_min) & (loc['latitude'] <= lat_max)
    )
    loc = loc[in_bbox].copy()

    col_w = (lon_max - lon_min) / GRID_COLS
    row_h = (lat_max - lat_min) / GRID_ROWS
    col = np.minimum(((loc['longitude'] - lon_min) / col_w).astype(int), GRID_COLS - 1)
    row = np.minimum(((loc['latitude'] - lat_min) / row_h).astype(int), GRID_ROWS - 1)
    loc['region_id'] = row * GRID_COLS + col

    base_to_region = dict(zip(loc['base_id'], loc['region_id']))

    poi_cols = [c for c in loc.columns if c.startswith('poi_')]
    region_poi_sum = loc.groupby('region_id')[poi_cols].sum()

    region_coords = []
    for r in range(NUM_REGIONS):
        row_idx, col_idx = divmod(r, GRID_COLS)
        lon_c = lon_min + (col_idx + 0.5) * col_w
        lat_c = lat_min + (row_idx + 0.5) * row_h
        region_coords.append((lon_c, lat_c))

    poi_fracs = np.zeros((NUM_REGIONS, len(POI_COLS)), dtype=np.float64)
    for r in region_poi_sum.index:
        counts = region_poi_sum.loc[r].values.astype(np.float64)
        total = counts.sum()
        if total > 0:
            poi_fracs[r] = counts / total

    bounds = dict(LON_MIN=float(lon_min), LON_MAX=float(lon_max),
                  LAT_MIN=float(lat_min), LAT_MAX=float(lat_max))
    return base_to_region, region_coords, poi_fracs, bounds


def load_events(records_dir, base_to_region):
    frames = []
    for path in sorted(glob.glob(os.path.join(records_dir, '*.csv'))):
        df = pd.read_csv(path, usecols=['user_id', 'time', 'location'], dtype={'time': str})
        df['region_id'] = df['location'].map(base_to_region)
        df = df.dropna(subset=['region_id'])
        df['region_id'] = df['region_id'].astype(int)
        df['date'] = df['time'].str.slice(0, 8)
        df['hour'] = df['time'].str.slice(8, 10).astype(int)
        frames.append(df[['user_id', 'date', 'hour', 'region_id']])
    return pd.concat(frames, ignore_index=True)


def build_user_day_sequences(events):
    """Majority region per observed hour, then forward/back-fill. Returns list of
    (user_id, date, seq[24], n_observed)."""
    groups = defaultdict(lambda: defaultdict(Counter))
    for user_id, date, hour, region_id in events.itertuples(index=False):
        groups[(user_id, date)][hour][region_id] += 1

    results = []
    for (user_id, date), hour_counters in groups.items():
        observed = {h: counter.most_common(1)[0][0] for h, counter in hour_counters.items()}
        n_observed = len(observed)
        if n_observed < MIN_OBSERVED_HOURS:
            continue

        seq = [None] * SEQ_LEN
        for h, r in observed.items():
            seq[h] = r

        last = None
        for h in range(SEQ_LEN):
            if seq[h] is None:
                seq[h] = last
            else:
                last = seq[h]
        first = next(r for r in seq if r is not None)
        for h in range(SEQ_LEN):
            if seq[h] is None:
                seq[h] = first

        results.append((user_id, date, seq, n_observed))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', required=True, choices=['shanghai', 'nanchang'])
    args = parser.parse_args()

    city_dir = os.path.join(HERE, args.city)
    out_dir = os.path.join(city_dir, 'derived')
    os.makedirs(out_dir, exist_ok=True)

    print('[%s] building grid from location.csv...' % args.city)
    base_to_region, region_coords, poi_fracs, bounds = build_grid(os.path.join(city_dir, 'location.csv'))
    print('  bbox:', bounds)
    print('  base stations mapped:', len(base_to_region))

    print('[%s] loading app_usage_records...' % args.city)
    events = load_events(os.path.join(city_dir, 'app_usage_records'), base_to_region)
    print('  events kept (in-bbox):', len(events))

    print('[%s] building per-user-day hourly sequences (impute + min-coverage filter)...' % args.city)
    seqs = build_user_day_sequences(events)
    n_days_total = len(events.groupby(['user_id', 'date']).size())
    print('  user-days with >=%d observed hours: %d / %d total user-days' %
          (MIN_OBSERVED_HOURS, len(seqs), n_days_total))

    rng = np.random.RandomState(SEED)
    users = sorted(set(u for u, _, _, _ in seqs))
    rng.shuffle(users)
    n_test_users = max(1, int(len(users) * TEST_USER_FRAC))
    test_users = set(users[:n_test_users])
    train_users = set(users[n_test_users:])
    print('  users: %d train, %d test' % (len(train_users), len(test_users)))

    def region_poi_str(region_id):
        top3 = np.argsort(-poi_fracs[region_id])[:3]
        return '|'.join(POI_COLS[i] for i in top3 if poi_fracs[region_id][i] > 0)

    loc_path = os.path.join(out_dir, 'location.csv')
    with open(loc_path, 'w') as f:
        f.write('region_id,lon_center,lat_center,' + ','.join(POI_COLS) + '\n')
        for r in range(NUM_REGIONS):
            lon_c, lat_c = region_coords[r]
            fracs = ','.join('%.3g' % v for v in poi_fracs[r])
            f.write('%d,%s,%s,%s\n' % (r, lon_c, lat_c, fracs))

    for split_name, split_users, path in [
        ('train', train_users, os.path.join(out_dir, 'mobility_train.csv')),
        ('test', test_users, os.path.join(out_dir, 'mobility_test.csv')),
    ]:
        n_rows = 0
        with open(path, 'w') as f:
            f.write('user_id,date,time,region_id,region_poi\n')
            for user_id, date, seq, n_observed in seqs:
                if user_id not in split_users:
                    continue
                for h, r in enumerate(seq):
                    f.write('%s,%s,%d,%d,%s\n' % (user_id, date, h, r, region_poi_str(r)))
                    n_rows += 1
        print('  wrote %s: %d rows -> %s' % (split_name, n_rows, path))

    print('[%s] done. bounds for config.py: %s' % (args.city, bounds))


if __name__ == '__main__':
    main()
