"""Pairwise (K=2) certification stream used by E2 and both E4 drivers.

One pair (j, k) plus a total-mass process is streamed.
 * pool="tracked" (E2, synthetic E4; T = T_j + T_k): the hybrid rule on
   `MultiCampaignCumulativeShareCS` (numerators j, k plus a shared denominator, beta = alpha/3).
 * pool="full" (Criteo E4; T includes the other campaigns and delta > 0): the estimand is the
   full-pool gap, so dropping untracked mass would be anti-conservative
   (`certify.UntrackedMassError`). The hybrid rule is unavailable, so this uses the
   direct-numerator rule with an explicit "other" process
   (`DirectNumeratorCS(include_other=True)`: j, k, other at alpha/3). This is the one
   exception to "hybrid everywhere" and is recorded in every output.
 * method "legacy_ratio_box" (configurations A/C): the ratio-box rule on
   `MultiCampaignCumulativeShareCS` for both pools.
"""
from __future__ import annotations

import numpy as np

from .certify import Certification, active_cert_method, box_rule, direct_robust_rule, robust_rule, snapshot
from .cs import DirectNumeratorCS, MultiCampaignCumulativeShareCS


class PairStream:
    def __init__(self, alpha: float, B: float, pool: str = "tracked", method: str | None = None):
        if pool not in ("tracked", "full"):
            raise ValueError("pool must be 'tracked' or 'full'")
        self.pool = pool
        self.method = active_cert_method() if method is None else method
        self.kind = "direct_other" if (pool == "full" and self.method == "hybrid") else self.method
        if self.kind == "direct_other":
            self.cs = DirectNumeratorCS(2, alpha, B, include_other=True)
        else:
            self.cs = MultiCampaignCumulativeShareCS(2, alpha, B)
        self.boundary = self.cs.boundary

    @property
    def label(self) -> str:
        return {"direct_other": "direct numerator + other bucket (full pool)",
                "hybrid": "hybrid (tracked pool)", "legacy_ratio_box": "legacy ratio box"}[self.kind]

    def update(self, Tj: float, Tk: float, T: float) -> None:
        self.cs.update(np.array([Tj, Tk]), T)

    def snap(self):
        if self.kind == "direct_other":
            LN, UN = self.cs.numerator_boxes()
            return ("direct_other", np.asarray(LN), np.asarray(UN), self.cs.other_box())
        return ("box", snapshot(self.cs))

    def rule(self, s, Gamma: float, delta: float) -> Certification:
        if s[0] == "direct_other":
            return direct_robust_rule(s[1], s[2], Gamma, delta, 0, 1, other=s[3], pool="full",
                                      untracked_mass=bool(self.cs.untracked_mass_seen))
        return box_rule(s[1], Gamma, delta, 0, 1, method=self.method)
