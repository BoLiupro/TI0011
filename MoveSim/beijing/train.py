import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
import data
import geo
import rollout
from discriminator import Discriminator
from generator import MoveSimGenerator


def log_print(log_file, msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()


def generate_pool(generator, num, batch_size, device, start_dist):
    """Unconditioned generation of `num` sequences, in batches."""
    out = []
    remaining = num
    while remaining > 0:
        b = min(batch_size, remaining)
        samples = generator.sample(batch_size=b, device=device, start_dist=start_dist, sample=True)
        out.append(samples)
        remaining -= b
    return torch.cat(out, dim=0)


def train_discriminator(discriminator, opt, real_pool, fake_pool, batch_size, epochs):
    n = min(real_pool.size(0), fake_pool.size(0))
    real_pool = real_pool[:n]
    fake_pool = fake_pool[:n]
    x = torch.cat([real_pool, fake_pool], dim=0)
    y = torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)], dim=0).to(x.device)

    losses = []
    for _ in range(epochs):
        perm = torch.randperm(x.size(0), device=x.device)
        x_s, y_s = x[perm], y[perm]
        for i in range(0, x_s.size(0) - batch_size + 1, batch_size):
            xb = x_s[i:i + batch_size]
            yb = y_s[i:i + batch_size]
            logits = discriminator(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float('nan')


def pretrain_generator_epoch(generator, opt, loader, device):
    losses = []
    for batch in loader:
        batch = batch.to(device)
        loss = generator.pretrain_loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def distance_loss(xy_coords_t, samples, scale):
    """samples: LongTensor [B, seq_len]. xy_coords_t: FloatTensor [vocab_size, 2] local meters.
    Mean squared distance between consecutive GENERATED locations, normalized by the study
    area's spatial range scale**2 (MoveSim's auxiliary physical-plausibility regularizer --
    raw meters**2 would be ~1e6-1e7, dwarfing the REINFORCE term; normalizing keeps it on a
    comparable, dimensionless scale)."""
    pts = xy_coords_t[samples]           # [B, seq_len, 2]
    diff = pts[:, 1:] - pts[:, :-1]
    return (diff ** 2).sum(dim=-1).mean() / (scale ** 2)


def adversarial_generator_step(generator, discriminator, opt, batch_size, device, start_dist, xy_coords_t, scale):
    samples = generator.sample(batch_size=batch_size, device=device, start_dist=start_dist, sample=True)
    rewards = rollout.get_reward(generator, discriminator, samples, rollout_num=config.ADV_ROLLOUT_NUM)
    log_probs = generator.sampled_log_probs(samples)          # [B, seq_len-1]
    pg_loss = -(log_probs * rewards[:, :-1]).sum(dim=1).mean()
    d_loss_term = distance_loss(xy_coords_t, samples, scale)
    loss = pg_loss + config.DLOSS_ALPHA * d_loss_term
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item(), rewards.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--pre_epoch_g', type=int, default=config.PRE_EPOCH_G)
    parser.add_argument('--d_pretrain_rounds', type=int, default=config.D_PRETRAIN_ROUNDS)
    parser.add_argument('--adv_total_rounds', type=int, default=config.ADV_TOTAL_ROUNDS)
    parser.add_argument('--generated_num', type=int, default=config.GENERATED_NUM)
    args = parser.parse_args()

    if args.smoke_test:
        args.pre_epoch_g = 1
        args.d_pretrain_rounds = 1
        args.adv_total_rounds = 2
        args.generated_num = 256

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    train_seqs, _train_user_ids = data.load_sequences(config.TRAIN_FILE)
    train_loader = data.make_loader(train_seqs, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    train_tensor = torch.from_numpy(train_seqs).long().to(device)

    coords = geo.load_region_coords()
    M1 = data.build_transition_matrix(train_seqs)
    M2 = geo.pairwise_distance_matrix(coords).astype(np.float32)
    start_dist = data.start_location_distribution(train_seqs)
    xy_coords_t = torch.from_numpy(geo.equirectangular_xy_meters(coords)).float().to(device)
    scale = geo.range_scale()

    generator = MoveSimGenerator(M1, M2).to(device)
    discriminator = Discriminator().to(device)

    g_pretrain_opt = torch.optim.Adam(generator.parameters(), lr=config.G_LR)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=config.D_LR)
    g_adv_opt = torch.optim.Adam(generator.parameters(), lr=config.ADV_G_LR)

    os.makedirs(config.SAVE_DIR, exist_ok=True)
    log_file = open(config.TRAIN_LOG, 'w')

    t0 = time.time()
    log_print(log_file, 'Stage 1: MLE pretraining generator...')
    for epoch in range(args.pre_epoch_g):
        loss = pretrain_generator_epoch(generator, g_pretrain_opt, train_loader, device)
        log_print(log_file, 'pretrain_g epoch %d loss %.4f (%.1fs elapsed)' % (epoch, loss, time.time() - t0))

    torch.save(generator.state_dict(), config.GENERATOR_PRETRAIN_CKPT)
    log_print(log_file, 'Saved MLE-pretrained generator to %s' % config.GENERATOR_PRETRAIN_CKPT)

    log_print(log_file, 'Stage 2: pretraining discriminator...')
    for rnd in range(args.d_pretrain_rounds):
        generator.eval()
        fake_pool = generate_pool(generator, args.generated_num, config.BATCH_SIZE, device, start_dist)
        generator.train()
        idx = torch.randint(0, train_tensor.size(0), (args.generated_num,), device=device)
        real_pool = train_tensor[idx]
        d_loss = train_discriminator(discriminator, d_opt, real_pool, fake_pool,
                                      config.BATCH_SIZE, config.D_PRETRAIN_EPOCHS_PER_ROUND)
        log_print(log_file, 'pretrain_d round %d d_loss %.4f (%.1fs elapsed)' % (rnd, d_loss, time.time() - t0))

    log_print(log_file, 'Stage 3: adversarial training...')
    for rnd in range(args.adv_total_rounds):
        generator.train()
        g_loss, mean_reward = adversarial_generator_step(generator, discriminator, g_adv_opt,
                                                           config.BATCH_SIZE, device, start_dist, xy_coords_t, scale)

        if rnd % config.ADV_D_REFRESH_EVERY == 0:
            for _ in range(config.ADV_D_ROUNDS):
                generator.eval()
                fake_pool = generate_pool(generator, args.generated_num, config.BATCH_SIZE, device, start_dist)
                generator.train()
                idx = torch.randint(0, train_tensor.size(0), (args.generated_num,), device=device)
                real_pool = train_tensor[idx]
                d_loss = train_discriminator(discriminator, d_opt, real_pool, fake_pool,
                                              config.BATCH_SIZE, config.ADV_D_EPOCHS_PER_ROUND)
        else:
            d_loss = float('nan')

        log_print(log_file, 'adv round %d g_loss %.4f mean_reward %.4f d_loss %.4f (%.1fs elapsed)' %
                   (rnd, g_loss, mean_reward, d_loss, time.time() - t0))

    torch.save(generator.state_dict(), config.GENERATOR_CKPT)
    torch.save(discriminator.state_dict(), config.DISCRIMINATOR_CKPT)
    log_print(log_file, 'Saved checkpoints to %s' % config.SAVE_DIR)
    log_file.close()


if __name__ == '__main__':
    main()
