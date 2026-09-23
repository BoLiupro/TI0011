import argparse

import numpy as np
import torch

import config
import data
import geo
from generator import MoveSimGenerator


def ratio_metrics(errors, uids, mean_map, max_map):
    """errors: np.array [N, K] per-sequence, per-hour displacement errors (meters). uids: array
    [N] of the user each row belongs to. Divides every element of a row by that user's pooled
    mean/max real displacement (from geo.user_displacement_stats); users absent from the maps
    (empty/all-zero real displacement) are skipped."""
    ratio_mean, ratio_max = [], []
    for row, uid in zip(errors, uids):
        m = mean_map.get(uid)
        mx = max_map.get(uid)
        if m:
            ratio_mean.extend((row / m).tolist())
        if mx:
            ratio_max.extend((row / mx).tolist())
    return (float(np.mean(ratio_mean)) if ratio_mean else float('nan'),
            float(np.mean(ratio_max)) if ratio_max else float('nan'))


def next_location_prediction(generator, test_seqs_t, test_user_ids, coords, mean_map, max_map,
                              device, batch_size=512):
    """Teacher-forced one-step-ahead prediction for hours 1..seq_len-1.
    Returns mean error (m), top-1 accuracy, and the two new per-user-normalized ratio
    metrics, for the model and for a trivial repeat-last-location baseline."""
    model_err_chunks, baseline_err_chunks = [], []
    model_correct = baseline_correct = total = 0
    generator.eval()
    with torch.no_grad():
        for i in range(0, test_seqs_t.size(0), batch_size):
            x = test_seqs_t[i:i + batch_size].to(device)
            log_probs = generator.teacher_forced_logits(x)  # [B, seq_len-1, vocab]
            pred = log_probs.argmax(dim=-1)

            pred_np = pred.cpu().numpy()
            true_np = x[:, 1:].cpu().numpy()
            baseline_np = x[:, :-1].cpu().numpy()  # repeat-last-location baseline

            model_err_chunks.append(geo.region_distance_matrix_lookup(coords, pred_np, true_np))
            baseline_err_chunks.append(geo.region_distance_matrix_lookup(coords, baseline_np, true_np))
            model_correct += int((pred_np == true_np).sum())
            baseline_correct += int((baseline_np == true_np).sum())
            total += true_np.size

    model_errs = np.concatenate(model_err_chunks, axis=0)      # [N, seq_len-1]
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


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

    coords = geo.load_region_coords()
    scale = geo.range_scale()

    train_seqs, _train_user_ids = data.load_sequences(config.TRAIN_FILE)
    test_seqs, test_user_ids = data.load_sequences(config.TEST_FILE)
    test_seqs_t = torch.from_numpy(test_seqs).long()

    mean_map, max_map = geo.user_displacement_stats(coords, test_seqs, test_user_ids)

    M1 = data.build_transition_matrix(train_seqs)
    M2 = geo.pairwise_distance_matrix(coords).astype(np.float32)

    pretrain_generator = MoveSimGenerator(M1, M2).to(device)
    pretrain_generator.load_state_dict(torch.load(config.GENERATOR_PRETRAIN_CKPT, map_location=device))

    adv_generator = MoveSimGenerator(M1, M2).to(device)
    adv_generator.load_state_dict(torch.load(config.GENERATOR_CKPT, map_location=device))

    pretrain_result = next_location_prediction(pretrain_generator, test_seqs_t, test_user_ids,
                                                coords, mean_map, max_map, device)
    adv_result = next_location_prediction(adv_generator, test_seqs_t, test_user_ids,
                                           coords, mean_map, max_map, device)

    lines = []
    lines.append('range_scale (sqrt(width*height), m): %.2f' % scale)
    lines.append('')
    lines.append('[Next location prediction] (teacher-forced, hours 1..23)')
    lines.append('  MLE-pretrained generator        mean_error_m: %.2f  ratio_to_range: %.4f  top1_acc: %.4f  '
                  'disp/mean_disp: %.4f  disp/max_disp: %.4f' %
                  (pretrain_result['model_error_m'], pretrain_result['model_error_m'] / scale,
                   pretrain_result['model_acc'], pretrain_result['model_ratio_mean_disp'],
                   pretrain_result['model_ratio_max_disp']))
    lines.append('  adversarially-trained generator mean_error_m: %.2f  ratio_to_range: %.4f  top1_acc: %.4f  '
                  'disp/mean_disp: %.4f  disp/max_disp: %.4f' %
                  (adv_result['model_error_m'], adv_result['model_error_m'] / scale,
                   adv_result['model_acc'], adv_result['model_ratio_mean_disp'],
                   adv_result['model_ratio_max_disp']))
    lines.append('  repeat-last-location baseline   mean_error_m: %.2f  ratio_to_range: %.4f  top1_acc: %.4f  '
                  'disp/mean_disp: %.4f  disp/max_disp: %.4f' %
                  (pretrain_result['baseline_error_m'], pretrain_result['baseline_error_m'] / scale,
                   pretrain_result['baseline_acc'], pretrain_result['baseline_ratio_mean_disp'],
                   pretrain_result['baseline_ratio_max_disp']))

    report = '\n'.join(lines)
    print(report)
    with open(config.EVAL_RESULTS, 'w') as f:
        f.write(report + '\n')


if __name__ == '__main__':
    main()
