"""Evaluate AgentMove's next-location predictions on the app_usage datasets with the *same*
metrics SeqGAN/app_usage_<city>/evaluate.py reports (mean_error_m, ratio_to_range, top1_acc),
instead of AgentMove's own venue-hash-string top1/3/5/MRR/MAP/NDCG evaluator (evaluate/evaluations.py),
which assumes 24-char hex Foursquare venue ids and doesn't apply to grid-region ids.

Reads the per-trajectory prediction JSONs Agent.predict() writes to
results/<exp_name>/<city_name>/agentmove/<model_name>/<prompt_type>/*.json, each containing
{'prediction': [...], 'true': <ground truth region_id>, ...} (see agent.py:Agent.predict).
Takes the top-1 predicted region id and compares it to ground truth using that city's own
SeqGAN geo.py (haversine, range_scale) for exact metric parity with the SeqGAN evaluation.

This script only reads existing prediction JSONs -- it never calls the LLM itself, so it is
safe to run before the API key is configured (there just won't be any predictions to read yet).
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, '..', '..')


def _load_city_geo(city):
    seqgan_dir = os.path.join(REPO_ROOT, 'SeqGAN', 'app_usage_%s' % city)
    sys.path.insert(0, seqgan_dir)
    for mod in ('config', 'geo'):
        sys.modules.pop(mod, None)
    import config as city_config
    import geo as city_geo
    coords = city_geo.load_region_coords()
    scale = city_geo.range_scale()
    return city_geo, city_config, coords, scale


def _extract_top1(prediction):
    """AgentMove's `prediction` field is a list (extract_json's output). Take the first element
    it can parse as a region id."""
    if not isinstance(prediction, list):
        return None
    for p in prediction:
        try:
            return int(p)
        except (TypeError, ValueError):
            continue
    return None


def evaluate(city, results_dir):
    city_geo, city_config, coords, scale = _load_city_geo(city)
    num_regions = city_config.NUM_REGIONS

    files = sorted(glob.glob(os.path.join(results_dir, '*.json')))
    total = 0
    valid = 0
    correct = 0
    errors_m = []

    for path in files:
        with open(path) as f:
            entry = json.load(f)
        total += 1
        true_region = entry.get('true')
        pred_region = _extract_top1(entry.get('prediction'))
        try:
            true_region = int(true_region)
        except (TypeError, ValueError):
            continue
        if pred_region is None or not (0 <= pred_region < num_regions) or not (0 <= true_region < num_regions):
            continue

        valid += 1
        if pred_region == true_region:
            correct += 1
        lat1, lon1 = coords[pred_region][1], coords[pred_region][0]
        lat2, lon2 = coords[true_region][1], coords[true_region][0]
        errors_m.append(city_geo.haversine(lat1, lon1, lat2, lon2))

    mean_error_m = float(sum(errors_m) / len(errors_m)) if errors_m else float('nan')
    top1_acc = correct / valid if valid else float('nan')

    return {
        'total_files': total,
        'valid_predictions': valid,
        'mean_error_m': mean_error_m,
        'ratio_to_range': mean_error_m / scale if errors_m else float('nan'),
        'top1_acc': top1_acc,
        'range_scale': scale,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', required=True, choices=['shanghai', 'nanchang'])
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--city_name', default=None, help='AgentMove --city_name used for the run (defaults to app_usage_<city>)')
    parser.add_argument('--model_name', default='qwen2.5-7b')
    parser.add_argument('--prompt_type', default='agent_move_v6')
    parser.add_argument('--results_dir', default=None, help='override auto-located results dir')
    args = parser.parse_args()

    city_name = args.city_name or ('app_usage_%s' % args.city)
    results_dir = args.results_dir or os.path.join(
        REPO_ROOT, 'AgentMove', 'results', args.exp_name, city_name, 'agentmove', args.model_name, args.prompt_type)

    print('reading predictions from: %s' % results_dir)
    result = evaluate(args.city, results_dir)
    print('[%s] total_files=%d valid_predictions=%d' % (args.city, result['total_files'], result['valid_predictions']))
    print('  mean_error_m: %.2f  ratio_to_range: %.4f  top1_acc: %.4f  (range_scale=%.2f)' %
          (result['mean_error_m'], result['ratio_to_range'], result['top1_acc'], result['range_scale']))
