import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class MoveSimGenerator(nn.Module):
    """Attention-gated Markov transition model (port of the MoveSim paper's ATGenerator).

    Embeds (location, hour-of-day), runs 2 stacked bidirectional MultiheadAttention layers,
    then gates the vocab-sized logits with two domain-prior matrices looked up by current
    location: M1 (empirical bigram transition-count matrix) and M2 (pairwise geographic
    distance matrix). M1/M2 are kept as GPU-resident buffers (not numpy, unlike the reference
    implementation) so every lookup stays on-device.
    """

    def __init__(self, M1, M2, vocab_size=config.VOCAB_SIZE, loc_emb_dim=config.LOC_EMB_DIM,
                 tim_emb_dim=config.TIM_EMB_DIM, hidden_dim=config.HIDDEN_DIM):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = loc_emb_dim + tim_emb_dim
        self.hidden_dim = hidden_dim

        self.register_buffer('M1', torch.as_tensor(M1, dtype=torch.float32))
        self.register_buffer('M2', torch.as_tensor(M2, dtype=torch.float32))

        self.loc_embedding = nn.Embedding(vocab_size, loc_emb_dim)
        self.tim_embedding = nn.Embedding(24, tim_emb_dim)

        self.Q = nn.Linear(self.embedding_dim, hidden_dim)
        self.K = nn.Linear(self.embedding_dim, hidden_dim)
        self.V = nn.Linear(self.embedding_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)

        self.Q2 = nn.Linear(hidden_dim, hidden_dim)
        self.K2 = nn.Linear(hidden_dim, hidden_dim)
        self.V2 = nn.Linear(hidden_dim, hidden_dim)
        self.attn2 = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)

        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.linear_mat1 = nn.Linear(vocab_size, hidden_dim)
        self.linear_mat1_2 = nn.Linear(hidden_dim, vocab_size)
        self.linear_mat2 = nn.Linear(vocab_size, hidden_dim)
        self.linear_mat2_2 = nn.Linear(hidden_dim, vocab_size)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def _encode(self, x_loc, x_tim):
        lemb = self.loc_embedding(x_loc)
        temb = self.tim_embedding(x_tim)
        x = torch.cat([lemb, temb], dim=-1)  # [B, L, embedding_dim]

        q, k, v = F.relu(self.Q(x)), F.relu(self.K(x)), F.relu(self.V(x))
        x, _ = self.attn(q, k, v)

        q, k, v = F.relu(self.Q2(x)), F.relu(self.K2(x)), F.relu(self.V2(x))
        x, _ = self.attn2(q, k, v)
        return x  # [B, L, hidden_dim]

    def _head_and_gate(self, x_loc, h):
        """h: [B, L, hidden_dim] encoder output. Returns pre-softmax scores [B, L, vocab_size]."""
        x = F.relu(self.linear(h))       # [B, L, vocab_size]
        mat1 = self.M1[x_loc]            # [B, L, vocab_size], GPU-resident lookup
        mat2 = self.M2[x_loc]
        mat1 = F.normalize(torch.sigmoid(self.linear_mat1_2(F.relu(self.linear_mat1(mat1)))), dim=-1)
        mat2 = F.normalize(torch.sigmoid(self.linear_mat2_2(F.relu(self.linear_mat2(mat2)))), dim=-1)
        return x + x * mat1 + x * mat2

    def forward(self, x_loc, x_tim):
        """x_loc, x_tim: LongTensor [B, L]. Returns log-probs [B, L, vocab_size], bidirectional
        over the given window (no causal mask needed: callers never include the prediction
        target in x_loc/x_tim)."""
        h = self._encode(x_loc, x_tim)
        scores = self._head_and_gate(x_loc, h)
        return F.log_softmax(scores, dim=-1)

    def teacher_forced_logits(self, seq):
        """seq: LongTensor [B, seq_len] true region ids for all hours 0..seq_len-1.
        Returns log-probs [B, seq_len-1, vocab_size]: position i is the prediction for hour
        i+1, given true hours 0..seq_len-2 (bidirectionally attended as a single window)."""
        x_loc = seq[:, :-1]
        B, L = x_loc.shape
        x_tim = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)
        return self.forward(x_loc, x_tim)

    def pretrain_loss(self, seq):
        log_probs = self.teacher_forced_logits(seq)
        target = seq[:, 1:]
        return F.nll_loss(log_probs.reshape(-1, self.vocab_size), target.reshape(-1))

    def step(self, x_loc, x_tim):
        """x_loc, x_tim: LongTensor [B, 1] (or [N, 1] flattened). Returns softmax probs
        [B, vocab_size] for the location one hour after (x_loc, x_tim). Memoryless: matches
        the reference implementation's single-position generation/scoring."""
        h = self._encode(x_loc, x_tim)
        scores = self._head_and_gate(x_loc, h)
        return F.softmax(scores.squeeze(1), dim=-1)

    @torch.no_grad()
    def sample(self, batch_size=None, seq_len=config.SEQ_LEN, prefix=None, device=None,
               start_dist=None, sample=True):
        """Autoregressively generate a full sequence of length seq_len, one location at a
        time via `step` (memoryless, matching the reference implementation's generation).
        prefix: optional LongTensor [B, k] real region ids to condition on (k < seq_len).
        Returns LongTensor [B, seq_len].
        """
        if prefix is not None:
            tokens = prefix.clone()
            batch_size = prefix.size(0)
            device = prefix.device
        elif start_dist is not None:
            dist_t = torch.as_tensor(start_dist, dtype=torch.float32, device=device)
            tokens = torch.multinomial(dist_t, batch_size, replacement=True).view(batch_size, 1)
        else:
            tokens = torch.randint(0, self.vocab_size, (batch_size, 1), device=device)

        for i in range(tokens.size(1), seq_len):
            cur_loc = tokens[:, -1:]
            cur_tim = torch.full((batch_size, 1), i - 1, dtype=torch.long, device=device)
            probs = self.step(cur_loc, cur_tim)
            next_tok = torch.multinomial(probs, 1) if sample else probs.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
        return tokens

    def sampled_log_probs(self, x):
        """x: LongTensor [B, seq_len], a self-consistent sequence produced by `sample`.
        Returns log pi(x[:,i] | x[:,i-1]) for positions i=1..seq_len-1, shape [B, seq_len-1]
        (scored the same memoryless way the sequence was generated, via `step`)."""
        B, L = x.shape
        src = x[:, :-1]
        tim = torch.arange(L - 1, device=x.device).unsqueeze(0).expand(B, -1)
        tgt = x[:, 1:]

        flat_loc = src.reshape(-1, 1)
        flat_tim = tim.reshape(-1, 1)
        probs = self.step(flat_loc, flat_tim)
        log_probs = torch.log(probs.clamp_min(1e-12)).reshape(B, L - 1, self.vocab_size)
        return log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
