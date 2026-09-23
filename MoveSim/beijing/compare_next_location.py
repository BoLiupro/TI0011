"""Combined next-location-prediction comparison: MoveSim (this dir's checkpoints) vs. SeqGAN
(SeqGAN/beijing's already-trained checkpoints, reused read-only -- no retraining), scored with
mean_error_m / ratio_to_range / top1_acc plus the two new per-user-normalized displacement
ratios (Displacement/Mean displacement, Displacement/Max displacement).
"""
import importlib
import os
import sys

import numpy as np
import torch

import config as ms_config
import data as ms_data
import geo as ms_geo
from generator import MoveSimGenerator
from evaluate import next_location_prediction as movesim_next_location_prediction, ratio_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
SEQGAN_DIR = os.path.abspath(os.path.join(HERE, '..', '..', 'SeqGAN', 'beijing'))

# SeqGAN's generator.py/config.py both use bare `import config`, which collides with this
# script's own `config` module (MoveSim's). Swap sys.modules['config'] out while importing
# SeqGAN's modules under isolation, then restore it.
_saved_config = sys.modules.pop('config', None)
_saved_generator = sys.modules.pop('generator', None)
sys.path.insert(0, SEQGAN_DIR)
try:
    seqgan_config = importlib.import_module('config')
    seqgan_generator_mod = importlib.import_module('generator')
    SeqGANGenerator = seqgan_generator_mod.Generator
finally:
    sys.path.remove(SEQGAN_DIR)
    sys.modules.pop('config', None)
    sys.modules.pop('generator', None)
    if _saved_config is not None:
        sys.modules['config'] = _saved_config
    if _saved_generator is not None:
        sys.modules['generator'] = _saved_generator


def seqgan_next_location_prediction(generator, test_seqs_t, test_user_ids, coords, mean_map, max_map,
                                     device, batch_size=512):
    """Same protocol as MoveSim's next_location_prediction, adapted to SeqGAN's Generator API
    (teacher_forced_logits returns all seq_len positions via a prepended BOS; slice off position 0)."""
    model_err_chunks, baseline_err_chunks = [], []
    model_correct = baseline_correct = total = 0
    generator.eval()
    with torch.no_grad():
        for i in range(0, test_seqs_t.size(0), batch_size):
            x = test_seqs_t[i:i + batch_size].to(device)
            logits = generator.teacher_forced_logits(x)  # [B, seq_len, vocab]
            pred = logits.argmax(dim=-1)[:, 1:]

            pred_np = pred.cpu().numpy()
            true_np = x[:, 1:].cpu().numpy()
            baseline_np = x[:, :-1].cpu().numpy()

            model_err_chunks.append(ms_geo.region_distance_matrix_lookup(coords, pred_np, true_np))
            baseline_err_chunks.append(ms_geo.region_distance_matrix_lookup(coords, baseline_np, true_np))
            model_correct += int((pred_np == true_np).sum())
            baseline_correct += int((baseline_np == true_np).sum())
            total += true_np.size

    model_errs = np.concatenate(model_err_chunks, axis=0)
    baseline_errs = np.concatenate(baseline_err_chunks, axis=0)

    model_ratio_mean, model_ratio_max = ratio_metrics(model_errs, test_user_ids, mean_map, max_map)
    baseline_ratio_mean, baseline_ratio_max = ratio_metrics(baseline_errs, test_user_ids, mean_map, max_map)

    return {
        'model_error_m': float(model_errs.mean()),
        'model_acc': model_correct / total,
        'model_ratio_mean_disp': model_ratio_mean,
        'model_ratio_max_disp': model_ratio_max,
        'baseline_error_m': float(baseline_errs.mean()),
        'baseline_acc': baseline_correct / total,
        'baseline_ratio_mean_disp': baseline_ratio_mean,
        'baseline_ratio_max_disp': baseline_ratio_max,
    }


def fmt_row(name, r, scale):
    return ('  %-32s mean_error_m: %8.2f  ratio_to_range: %.4f  top1_acc: %.4f  '
            'disp/mean_disp: %.4f  disp/max_disp: %.4f' %
            (name, r['model_error_m'], r['model_error_m'] / scale, r['model_acc'],
             r['model_ratio_mean_disp'], r['model_ratio_max_disp']))


def main():
    device = torch.device(ms_config.DEVICE if torch.cuda.is_available() else 'cpu')

    coords = ms_geo.load_region_coords()
    scale = ms_geo.range_scale()

    train_seqs, _train_user_ids = ms_data.load_sequences(ms_config.TRAIN_FILE)
    test_seqs, test_user_ids = ms_data.load_sequences(ms_config.TEST_FILE)
    test_seqs_t = torch.from_numpy(test_seqs).long()

    mean_map, max_map = ms_geo.user_displacement_stats(coords, test_seqs, test_user_ids)

    M1 = ms_data.build_transition_matrix(train_seqs)
    M2 = ms_geo.pairwise_distance_matrix(coords).astype(np.float32)

    ms_pretrain = MoveSimGenerator(M1, M2).to(device)
    ms_pretrain.load_state_dict(torch.load(ms_config.GENERATOR_PRETRAIN_CKPT, map_location=device))
    ms_adv = MoveSimGenerator(M1, M2).to(device)
    ms_adv.load_state_dict(torch.load(ms_config.GENERATOR_CKPT, map_location=device))

    sg_pretrain = SeqGANGenerator().to(device)
    sg_pretrain.load_state_dict(torch.load(seqgan_config.GENERATOR_PRETRAIN_CKPT, map_location=device))
    sg_adv = SeqGANGenerator().to(device)
    sg_adv.load_state_dict(torch.load(seqgan_config.GENERATOR_CKPT, map_location=device))

    ms_pretrain_result = movesim_next_location_prediction(ms_pretrain, test_seqs_t, test_user_ids,
                                                            coords, mean_map, max_map, device)
    ms_adv_result = movesim_next_location_prediction(ms_adv, test_seqs_t, test_user_ids,
                                                       coords, mean_map, max_map, device)
    sg_pretrain_result = seqgan_next_location_prediction(sg_pretrain, test_seqs_t, test_user_ids,
                                                           coords, mean_map, max_map, device)
    sg_adv_result = seqgan_next_location_prediction(sg_adv, test_seqs_t, test_user_ids,
                                                      coords, mean_map, max_map, device)

    lines = []
    lines.append('range_scale (sqrt(width*height), m): %.2f' % scale)
    lines.append('')
    lines.append('[Next location prediction] (teacher-forced, hours 1..23)')
    lines.append(fmt_row('MoveSim  MLE-pretrained generator', ms_pretrain_result, scale))
    lines.append(fmt_row('MoveSim  adversarially-trained generator', ms_adv_result, scale))
    lines.append(fmt_row('SeqGAN   MLE-pretrained generator', sg_pretrain_result, scale))
    lines.append(fmt_row('SeqGAN   adversarially-trained generator', sg_adv_result, scale))
    lines.append('  %-32s mean_error_m: %8.2f  ratio_to_range: %.4f  top1_acc: %.4f  '
                  'disp/mean_disp: %.4f  disp/max_disp: %.4f' %
                  ('repeat-last-location baseline', sg_pretrain_result['baseline_error_m'],
                   sg_pretrain_result['baseline_error_m'] / scale, sg_pretrain_result['baseline_acc'],
                   sg_pretrain_result['baseline_ratio_mean_disp'], sg_pretrain_result['baseline_ratio_max_disp']))

    report = '\n'.join(lines)
    print(report)
    with open(ms_config.COMPARE_RESULTS, 'w') as f:
        f.write(report + '\n')


if __name__ == '__main__':
    main()
