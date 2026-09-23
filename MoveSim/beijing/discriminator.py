import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class Highway(nn.Module):
    def __init__(self, size, num_layers=1, bias=-2.0):
        super().__init__()
        self.num_layers = num_layers
        self.lin = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        for g in self.gate:
            nn.init.constant_(g.bias, bias)

    def forward(self, x):
        for lin, gate in zip(self.lin, self.gate):
            g = F.relu(lin(x))
            t = torch.sigmoid(gate(x))
            x = t * g + (1.0 - t) * x
        return x


class Discriminator(nn.Module):
    """TextCNN-style sequence classifier: real vs generated location sequences.
    Filter sizes/counts ported verbatim from the MoveSim paper's reference code."""

    def __init__(self, vocab_size=config.VOCAB_SIZE, seq_len=config.SEQ_LEN,
                 emb_dim=config.DIS_EMB_DIM, filter_sizes=config.DIS_FILTER_SIZES,
                 num_filters=config.DIS_NUM_FILTERS, dropout_keep_prob=config.DIS_DROPOUT_KEEP_PROB):
        super().__init__()
        assert max(filter_sizes) <= seq_len
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, nf, kernel_size=fs) for fs, nf in zip(filter_sizes, num_filters)
        ])
        total_filters = sum(num_filters)
        self.highway = Highway(total_filters, num_layers=1)
        self.dropout = nn.Dropout(1.0 - dropout_keep_prob)
        self.out = nn.Linear(total_filters, 2)

    def forward(self, x):
        """x: LongTensor [B, seq_len] region ids. Returns logits [B, 2]."""
        emb = self.emb(x).transpose(1, 2)  # [B, emb_dim, seq_len]
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))                        # [B, nf, seq_len-fs+1]
            p = F.max_pool1d(c, c.size(-1)).squeeze(-1)   # [B, nf]
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        h = self.highway(h)
        h = self.dropout(h)
        return self.out(h)

    def prob_real(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)[:, 1]
