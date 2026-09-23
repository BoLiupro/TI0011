import torch

import config


@torch.no_grad()
def get_reward(generator, discriminator, samples, rollout_num=config.ADV_ROLLOUT_NUM):
    """samples: LongTensor [B, seq_len], a self-consistent sequence produced by `generator`.
    For each position, average the discriminator's real-probability over `rollout_num` Monte
    Carlo completions (given the true prefix up to that position). The last position uses the
    already-complete sample itself. Returns rewards FloatTensor [B, seq_len].
    """
    B, L = samples.shape
    device = samples.device
    rewards = torch.zeros(B, L, device=device)

    for _ in range(rollout_num):
        for given_num in range(1, L):
            prefix = samples[:, :given_num]
            completed = generator.generate(prefix=prefix, sample=True)
            ypred = discriminator.prob_real(completed)
            rewards[:, given_num - 1] += ypred
        ypred = discriminator.prob_real(samples)
        rewards[:, L - 1] += ypred

    rewards /= rollout_num
    return rewards
