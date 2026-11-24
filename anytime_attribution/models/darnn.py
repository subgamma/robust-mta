"""DARNN (attention-RNN) conversion model with attention-based per-touch credit.

Empirical: no identification guarantee; use only the empirical envelope Gamma_hat_sens, never Gamma*.
Same interface as the other models: `fit(train_journeys, n_campaigns, seed, negative_journeys)`
and `credit(journey) -> np.ndarray`.

A GRU encodes the touch sequence (campaign one-hot plus recency to conversion per
step); one additive-attention layer over the hidden states gives per-touch weights,
summed by campaign and normalised into the credit vector. This is a single-attention
simplification of Ji & Wang (2017), which uses input and temporal attention.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..journeys import Journey


class _DARNNNet(nn.Module):
    def __init__(self, n_campaigns: int, hidden_size: int = 16):
        super().__init__()
        self.n_campaigns = n_campaigns
        input_size = n_campaigns + 1
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        packed_out, _ = self.gru(x)
        scores = self.attn(packed_out).squeeze(-1)
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = (weights.unsqueeze(-1) * packed_out).sum(dim=1)
        logit = self.classifier(context).squeeze(-1)
        return logit, weights


def _journey_to_tensor(journey: Journey, n_campaigns: int, max_len: int) -> tuple[np.ndarray, int]:
    x = np.zeros((max_len, n_campaigns + 1), dtype=np.float32)
    touches = journey.touches[-max_len:]
    for i, t in enumerate(touches):
        x[i, t.campaign] = 1.0
        x[i, n_campaigns] = journey.conversion_time - t.timestamp
    return x, len(touches)


class DARNN:
    def __init__(self, hidden_size: int = 16, epochs: int = 5, lr: float = 1e-2, device: str = "cpu"):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.device = device

    def fit(
        self,
        train_journeys: list[Journey],
        n_campaigns: int,
        seed: int = 0,
        negative_journeys: Optional[list[Journey]] = None,
    ) -> None:
        if not negative_journeys:
            raise ValueError("DARNN.fit requires negative_journeys")
        torch.manual_seed(seed)
        self.n_campaigns = n_campaigns
        self.max_len = max(j.n_touches for j in train_journeys + negative_journeys)
        self.net = _DARNNNet(n_campaigns, self.hidden_size).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        journeys = train_journeys + negative_journeys
        labels = [1.0] * len(train_journeys) + [0.0] * len(negative_journeys)
        X, lengths = zip(*(_journey_to_tensor(j, n_campaigns, self.max_len) for j in journeys))
        X = torch.tensor(np.stack(X), device=self.device)
        lengths = torch.tensor(lengths, device=self.device)
        y = torch.tensor(labels, dtype=torch.float32, device=self.device)

        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits, _ = self.net(X, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        self.net.eval()

    def credit(self, journey: Journey) -> np.ndarray:
        x, length = _journey_to_tensor(journey, self.n_campaigns, max(journey.n_touches, self.max_len))
        with torch.no_grad():
            xt = torch.tensor(x[None, :], device=self.device)
            lt = torch.tensor([length], device=self.device)
            _, weights = self.net(xt, lt)
        weights = weights.cpu().numpy().reshape(-1)[:length]
        z = np.zeros(self.n_campaigns)
        for w, t in zip(weights, journey.touches[-length:]):
            z[t.campaign] += w
        total = z.sum()
        return z / total if total > 0 else z
