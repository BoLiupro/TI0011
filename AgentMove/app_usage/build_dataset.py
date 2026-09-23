"""Build AgentMove's (test_dictionary, true_locations) dict pair directly from the derived
app_usage mobility CSVs (data/app_usage/<city>/derived/{location,mobility_train,mobility_test}.csv),
bypassing processing/data.py's Foursquare/Nominatim-specific Dataset class entirely.

Region grid ids (from SeqGAN's preprocessing) are reused as venue_id, so predictions live in the
same coordinate space SeqGAN/app_usage_<city>/geo.py already knows how to score.

Schema matches processing/data.py's trajectory_split branch (agent.py's Agents/Agent classes only
ever call dataset.get_generated_datasets(), so this shim is a drop-in replacement for a real
Dataset instance):
  test_dictionary[user_id][traj_id] = {
      'historical_stays': [[hour, weekday, venue_category_name, venue_id, admin, subdistrict, poi, street], ...],
      'historical_pos':   [[lon, lat], ...],
      'historical_addr':  [[admin, subdistrict, poi, street], ...],
      'historical_stays_long': same shape as historical_stays, full history (used by SocialWorld/known_stays),
      'historical_addr_long':  same shape as historical_addr, full history,
      'context_stays':    [[hour, weekday, venue_category_name, venue_id, admin, subdistrict, poi, street], ...],
      'context_pos':      [[lon, lat], ...],
      'context_addr':     [[admin, subdistrict, poi, street], ...],
      'target_stay':      [hour, weekday, '<next_place_id>', '<next_place_address>'],
  }
  true_locations[user_id][traj_id] = {
      'ground_stay': venue_id, 'ground_pos': [lon, lat], 'ground_addr': [admin, subdistrict, poi, street],
  }

One AgentMove "trajectory" = one held-out day for a test user; historical_stays is built by
leave-one-day-out from that same user's other kept days (test users only -- app_usage's train/test
split is disjoint by user, mirroring SeqGAN's convention, so there's no separate pool of "train"
users to draw history from for a test user).
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, '..', '..', 'data', 'app_usage')

ADDR_PLACEHOLDER = ["", "", "", ""]


def _weekday(date_str):
    from datetime import datetime
    return datetime.strptime(date_str, '%Y%m%d').weekday()


def _dominant_category(region_poi):
    if not isinstance(region_poi, str) or not region_poi:
        return "Unknown"
    return region_poi.split('|')[0]


def _stay_row(hour, weekday, region_id, category):
    return [hour, weekday, category, region_id] + ADDR_PLACEHOLDER


class AppUsageDataset:
    """Drop-in replacement for processing/data.py's Dataset: only get_generated_datasets() is used
    by Agents/Agent/SocialWorld."""

    def __init__(self, city, max_test_days_per_user=3):
        city_dir = os.path.join(DATA_ROOT, city, 'derived')
        self.location = pd.read_csv(os.path.join(city_dir, 'location.csv')).set_index('region_id')
        mobility = pd.read_csv(os.path.join(city_dir, 'mobility_test.csv'), dtype={'date': str})
        self.max_test_days_per_user = max_test_days_per_user
        self.test_dictionary, self.true_locations = self._build(mobility)

    def _region_pos(self, region_id):
        row = self.location.loc[region_id]
        return [float(row['lon_center']), float(row['lat_center'])]

    def _region_category(self, region_id):
        row = self.location.loc[region_id]
        poi_cols = [c for c in self.location.columns if c not in ('lon_center', 'lat_center')]
        best = max(poi_cols, key=lambda c: row[c])
        return best if row[best] > 0 else "Unknown"

    def _day_stays(self, day_df):
        day_df = day_df.sort_values('time')
        stays = []
        for _, r in day_df.iterrows():
            hour = int(r['time'])
            region_id = int(r['region_id'])
            weekday = _weekday(str(r['date']))
            stays.append(_stay_row(hour, weekday, region_id, _dominant_category(r['region_poi'])))
        return stays

    def _build(self, mobility):
        test_dictionary = {}
        true_locations = {}

        for user_id, user_df in mobility.groupby('user_id'):
            user_id = str(user_id)
            dates = sorted(user_df['date'].unique())
            days = {d: self._day_stays(user_df[user_df['date'] == d]) for d in dates}

            test_dictionary[user_id] = {}
            true_locations[user_id] = {}

            target_dates = dates[-self.max_test_days_per_user:]
            for traj_idx, target_date in enumerate(target_dates):
                context_full = days[target_date]
                if len(context_full) < 4:
                    continue
                target_row = context_full[-1]
                context_stays = context_full[:-1]
                context_pos = [self._region_pos(r[3]) for r in context_stays]
                context_addr = [ADDR_PLACEHOLDER for _ in context_stays]

                historical_stays = []
                for d in dates:
                    if d == target_date:
                        continue
                    historical_stays.extend(days[d])
                if not historical_stays:
                    # AgentMove's Memory.user_profile_generation (models/personal_memory.py)
                    # crashes on an empty history -- skip test users/days with no other days.
                    continue
                historical_pos = [self._region_pos(r[3]) for r in historical_stays]
                historical_addr = [ADDR_PLACEHOLDER for _ in historical_stays]

                target_stay = target_row[:2] + ['<next_place_id>', '<next_place_address>']
                ground_region = target_row[3]

                traj_id = str(traj_idx)
                test_dictionary[user_id][traj_id] = {
                    'historical_stays': historical_stays[-15:],
                    'historical_pos': historical_pos[-15:],
                    'historical_addr': historical_addr[-15:],
                    'historical_stays_long': historical_stays,
                    'historical_addr_long': historical_addr,
                    'context_stays': context_stays,
                    'context_pos': context_pos,
                    'context_addr': context_addr,
                    'target_stay': target_stay,
                }
                true_locations[user_id][traj_id] = {
                    'ground_stay': ground_region,
                    'ground_pos': self._region_pos(ground_region),
                    'ground_addr': ADDR_PLACEHOLDER,
                }

        return test_dictionary, true_locations

    def get_generated_datasets(self):
        return self.test_dictionary, self.true_locations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', required=True, choices=['shanghai', 'nanchang'])
    args = parser.parse_args()

    ds = AppUsageDataset(args.city)
    test_dictionary, true_locations = ds.get_generated_datasets()

    n_users = len(test_dictionary)
    n_trajs = sum(len(v) for v in test_dictionary.values())
    print('[%s] users: %d, trajectories: %d' % (args.city, n_users, n_trajs))

    for user_id, trajs in test_dictionary.items():
        if not trajs:
            continue
        traj_id, sample = next(iter(trajs.items()))
        print('sample user_id=%s traj_id=%s' % (user_id, traj_id))
        print('  historical_stays (%d):' % len(sample['historical_stays']), sample['historical_stays'][:2])
        print('  context_stays (%d):' % len(sample['context_stays']), sample['context_stays'][:2])
        print('  target_stay:', sample['target_stay'])
        print('  ground:', true_locations[user_id][traj_id])
        break
