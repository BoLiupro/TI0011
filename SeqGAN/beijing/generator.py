import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class Generator(nn.Module):
    """GPT-style causal Transformer generator over location tokens.

    Vocab layout: ids 0..vocab_size-1 are real regions, id `bos` (=vocab_size) is a
    dedicated start token used only as model input, never as a prediction target.
    """

    def __init__(self, vocab_size=config.VOCAB_SIZE, bos=config.BOS, seq_len=config.SEQ_LEN,
                 emb_dim=config.EMB_DIM, n_layer=config.N_LAYER, n_head=config.N_HEAD,
                 ff_dim=config.FF_DIM, dropout=config.DROPOUT):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos = bos
        self.seq_len = seq_len
        self.emb_dim = emb_dim

        self.token_emb = nn.Embedding(vocab_size + 1, emb_dim)
        self.pos_emb = nn.Embedding(seq_len, emb_dim)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_head, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True, activation='gelu')
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

        self._mask_cache = {}

    def _causal_mask(self, length, device):
        mask = self._mask_cache.get((length, device))
        if mask is None:
            mask = torch.triu(torch.full((length, length), float('-inf'), device=device), diagonal=1)
            self._mask_cache[(length, device)] = mask
        return mask

    def forward(self, input_ids):
        """input_ids: LongTensor [B, L] (L <= seq_len), may include BOS. Returns logits [B, L, vocab_size]."""
        B, L = input_ids.shape
        device = input_ids.device
        pos_ids = torch.arange(L, device=device)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids).unsqueeze(0)
        x = self.drop(x)
        mask = self._causal_mask(L, device)
        h = self.blocks(x, mask=mask)
        h = self.ln_f(h)
        return self.head(h)

    def _prepend_bos(self, x):
        B = x.size(0)
        bos_col = torch.full((B, 1), self.bos, dtype=torch.long, device=x.device)
        return torch.cat([bos_col, x[:, :-1]], dim=1)

    def teacher_forced_logits(self, x):
        """x: LongTensor [B, seq_len] of true region ids. Returns logits [B, seq_len, vocab_size]
        where position i is the prediction for x[:, i] given x[:, :i] (and BOS)."""
        input_ids = self._prepend_bos(x)
        return self.forward(input_ids)

    def pretrain_loss(self, x):
        logits = self.teacher_forced_logits(x)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), x.reshape(-1))

    @torch.no_grad()
    def generate(self, batch_size=None, device=None, prefix=None, sample=True, temperature=1.0):
        """Autoregressively generate a full sequence of length seq_len.
        prefix: optional LongTensor [B, k] real region ids to condition on (k < seq_len).
        Returns LongTensor [B, seq_len].
        """
        if prefix is not None:
            tokens = prefix.clone()
            batch_size = prefix.size(0)
            device = prefix.device
        else:
            tokens = torch.empty(batch_size, 0, dtype=torch.long, device=device)

        bos_col = torch.full((batch_size, 1), self.bos, dtype=torch.long, device=device)
        for step in range(tokens.size(1), self.seq_len):
            input_ids = torch.cat([bos_col, tokens], dim=1)
            logits = self.forward(input_ids)
            last_logits = logits[:, -1, :] / temperature
            if sample:
                probs = F.softmax(last_logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
            else:
                next_tok = last_logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
        return tokens

    def sampled_log_probs(self, x):
        """x: LongTensor [B, seq_len], a self-consistent generated (or real) sequence.
        Returns log pi(x[:,i] | x[:,:i]) for every position, shape [B, seq_len]."""
        logits = self.teacher_forced_logits(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, x.unsqueeze(-1)).squeeze(-1)
