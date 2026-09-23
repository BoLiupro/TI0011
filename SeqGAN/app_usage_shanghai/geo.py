import csv
import math

import numpy as np

import config


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two (lat, lon) points in degrees."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in meters. Inputs are numpy arrays (broadcastable), degrees."""
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_region_coords(path=config.LOCATION_FILE, num_regions=config.NUM_REGIONS):
    """Returns np.array[num_regions, 2] of (lat, lon) indexed by region_id."""
    coords = np.zeros((num_regions, 2), dtype=np.float64)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = int(row['region_id'])
            coords[rid, 0] = float(row['lat_center'])
            coords[rid, 1] = float(row['lon_center'])
    return coords


def range_scale():
    """sqrt(width_m * height_m) of the study-area bbox, the 'whole range' distance scale."""
    lat_mid = (config.LAT_MIN + config.LAT_MAX) / 2
    lon_mid = (config.LON_MIN + config.LON_MAX) / 2
    width_m = haversine(lat_mid, config.LON_MIN, lat_mid, config.LON_MAX)
    height_m = haversine(config.LAT_MIN, lon_mid, config.LAT_MAX, lon_mid)
    return math.sqrt(width_m * height_m)


def region_distance_matrix_lookup(coords, pred_ids, true_ids):
    """Elementwise haversine distance (meters) between coords[pred_ids] and coords[true_ids].
    pred_ids, true_ids: numpy int arrays of the same shape.
    """
    pred_ll = coords[pred_ids]
    true_ll = coords[true_ids]
    return haversine_np(pred_ll[..., 0], pred_ll[..., 1], true_ll[..., 0], true_ll[..., 1])


def pairwise_distance_matrix(coords):
    """coords: [N, 2] (lat, lon) degrees. Returns [N, N] haversine distance matrix in meters."""
    lat, lon = coords[:, 0], coords[:, 1]
    return haversine_np(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def equirectangular_xy_meters(coords):
    """coords: [N, 2] (lat, lon) degrees. Returns [N, 2] local flat-earth (x, y) meters,
    using the study-area bbox center as the origin (valid for this small area)."""
    r = 6371000.0
    lat0 = math.radians((config.LAT_MIN + config.LAT_MAX) / 2)
    lon0 = math.radians((config.LON_MIN + config.LON_MAX) / 2)
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    x = r * (lon - lon0) * math.cos(lat0)
    y = r * (lat - lat0)
    return np.stack([x, y], axis=1)


def user_displacement_stats(coords, sequences, user_ids):
    """For each unique user, pool the haversine distance between every pair of
    chronologically-consecutive real hours across ALL sequences (days) that user appears in.
    Returns (mean_map, max_map), each {user_id: displacement_m}. Users whose pooled
    displacement set is empty or all-zero are omitted (logged).
    """
    from collections import defaultdict as _defaultdict

    user_to_dists = _defaultdict(list)
    for seq, uid in zip(sequences, user_ids):
        lat = coords[seq, 0]
        lon = coords[seq, 1]
        d = haversine_np(lat[:-1], lon[:-1], lat[1:], lon[1:])
        user_to_dists[uid].extend(d.tolist())

    mean_map, max_map = {}, {}
    skipped = 0
    for uid, dists in user_to_dists.items():
        arr = np.array(dists, dtype=np.float64)
        if arr.size == 0 or not np.any(arr > 0):
            skipped += 1
            continue
        mean_map[uid] = float(arr.mean())
        max_map[uid] = float(arr.max())
    if skipped:
        print('user_displacement_stats: skipped %d/%d users with empty/all-zero displacement' %
              (skipped, len(user_to_dists)))
    return mean_map, max_map


if __name__ == '__main__':
    print('range_scale (m):', range_scale())
    coords = load_region_coords()
    print('coords shape:', coords.shape, coords[0], coords[499])
