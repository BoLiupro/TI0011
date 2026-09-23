import argparse

import numpy as np
import torch

import config
import data
import geo
from generator import Generator


def next_location_prediction(generator, test_seqs_t, coords, device, batch_size=512):
    """Teacher-forced one-step-ahead prediction for hours 1..seq_len-1.
    Returns dict with mean error (m) and top-1 accuracy for the model and for a
    trivial repeat-last-location baseline."""
    model_errors = []
    baseline_errors = []
    model_correct = 0
    baseline_correct = 0
    total = 0
    generator.eval()
    with torch.no_grad():
        for i in range(0, test_seqs_t.size(0), batch_size):
            x = test_seqs_t[i:i + batch_size].to(device)
            logits = generator.teacher_forced_logits(x)  # [B, seq_len, vocab]
            pred = logits.argmax(dim=-1)  # [B, seq_len]

            pred_np = pred[:, 1:].cpu().numpy()
            true_np = x[:, 1:].cpu().numpy()
            baseline_np = x[:, :-1].cpu().numpy()  # repeat-last-location baseline

            model_errors.append(geo.region_distance_matrix_lookup(coords, pred_np, true_np).reshape(-1))
            baseline_errors.append(geo.region_distance_matrix_lookup(coords, baseline_np, true_np).reshape(-1))
            model_correct += int((pred_np == true_np).sum())
            baseline_correct += int((baseline_np == true_np).sum())
            total += true_np.size

    model_errors = np.concatenate(model_errors)
    baseline_errors = np.concatenate(baseline_errors)
    return {
        'model_error_m': float(model_errors.mean()),
        'model_acc': model_correct / total,
        'baseline_error_m': float(baseline_errors.mean()),
        'baseline_acc': baseline_correct / total,
    }


def mobility_generation(generator, test_seqs_t, coords, device, k_samples=config.GEN_NUM_SAMPLES,
                         batch_size=256):
    """Condition on the true hour-0 region, freely generate the remaining hours k_samples times,
    average the per-hour distance to the real continuation."""
    generator.eval()
    all_errors = []
    with torch.no_grad():
        for i in range(0, test_seqs_t.size(0), batch_size):
            x = test_seqs_t[i:i + batch_size].to(device)
            prefix = x[:, :1]
            true_np = x[:, 1:].cpu().numpy()  # [B, seq_len-1]

            batch_errors = []
            for _ in range(k_samples):
                gen = generator.generate(prefix=prefix, sample=True)  # [B, seq_len]
                gen_np = gen[:, 1:].cpu().numpy()
                batch_errors.append(geo.region_distance_matrix_lookup(coords, gen_np, true_np))
            # [k_samples, B, seq_len-1] -> mean over samples and hours -> [B]
            batch_errors = np.stack(batch_errors, axis=0).mean(axis=(0, 2))
            all_errors.append(batch_errors)

    all_errors = np.concatenate(all_errors)
    return float(all_errors.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_samples', type=int, default=config.GEN_NUM_SAMPLES)
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

    coords = geo.load_region_coords()

    test_seqs, test_user_ids = data.load_sequences(config.TEST_FILE)
    test_seqs_t = torch.from_numpy(test_seqs).long()

    # Normalize errors against the population's own typical movement scale rather than the
    # study area's bounding-box size: average, over users, of each user's mean displacement
    # between consecutive real hours, counting only steps where they actually moved (users
    # who never move at all are excluded rather than diluting the average toward zero).
    scale = geo.avg_user_movement_m(coords, test_seqs, test_user_ids)

    pretrain_generator = Generator().to(device)
    pretrain_generator.load_state_dict(torch.load(config.GENERATOR_PRETRAIN_CKPT, map_location=device))

    adv_generator = Generator().to(device)
    adv_generator.load_state_dict(torch.load(config.GENERATOR_CKPT, map_location=device))

    # Next-location prediction is a supervised one-step-ahead task, so it is evaluated
    # with the MLE-pretrained generator (directly optimized for teacher-forced next-token
    # cross-entropy). The adversarially-trained generator is also reported for reference:
    # SeqGAN's policy-gradient objective optimizes discriminator-judged realism of full
    # sequences, not per-step argmax accuracy, so it commonly trades that accuracy away.
    pretrain_result = next_location_prediction(pretrain_generator, test_seqs_t, coords, device)
    adv_result = next_location_prediction(adv_generator, test_seqs_t, coords, device)

    # Mobility generation uses the fully adversarially-trained generator, since that stage
    # is what SeqGAN uses to make generated sequences realistic/diverse under the discriminator.
    gen_err = mobility_generation(adv_generator, test_seqs_t, coords, device, k_samples=args.k_samples)

    lines = []
    lines.append('avg_user_movement_m (per-user mean of non-stationary hourly steps, avg over users, '
                  'excluding never-moved users): %.2f' % scale)
    lines.append('')
    lines.append('[Next location prediction] (teacher-forced, hours 1..23)')
    lines.append('  MLE-pretrained generator      mean_error_m: %.2f  ratio_to_user_movement: %.4f  top1_acc: %.4f' %
                  (pretrain_result['model_error_m'], pretrain_result['model_error_m'] / scale,
                   pretrain_result['model_acc']))
    lines.append('  adversarially-trained generator mean_error_m: %.2f  ratio_to_user_movement: %.4f  top1_acc: %.4f' %
                  (adv_result['model_error_m'], adv_result['model_error_m'] / scale, adv_result['model_acc']))
    lines.append('')
    lines.append('[Mobility generation] (conditioned on hour-0, K=%d samples, hours 1..23, adversarially-trained generator)' %
                  args.k_samples)
    lines.append('  model  mean_error_m: %.2f  ratio_to_user_movement: %.4f' % (gen_err, gen_err / scale))

    report = '\n'.join(lines)
    print(report)
    with open(config.EVAL_RESULTS, 'w') as f:
        f.write(report + '\n')


if __name__ == '__main__':
    main()
