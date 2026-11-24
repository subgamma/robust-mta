"""E10: controlled multi-campaign synthetic benchmark.

Not observational evidence and not a causal model: it exercises the cross-campaign
within-journey regime that the single-campaign Criteo primary stream cannot. Constants
were fixed before any result was inspected; only journey-composition statistics (target:
45-55% of converting journeys with >= 2 distinct campaigns) were examined, never
certification, coverage or reversal outcomes.

Distinct objects:
 (1) latent conversion outcome: Bernoulli(sigmoid(logit(path))) per candidate path (`conversion_logit`)
 (2) latent full-observation credit: per-model credit on the full latent journey (`model_masses(..., keep=None)`)
 (3) thinned observed histories: `thin` (independent Bernoulli retention per touch, campaign-specific p_j)
 (4) observed attribution outputs: `model_masses(..., keep=mask)`
 (5) learned-model predictions: `LinearSHAPModel`, trained only on the independent warm-up dataset
Attribution shares are shares of model credit, not causal incremental effects.

Generator (K = 10 campaigns: 3 upper-funnel, 4 mid-funnel, 3 lower-funnel/retargeting):
 length: n_raw ~ truncated Zipf(a=LENGTH_EXP) on 1..40; the latent journey keeps its 10 most
         recent touches (cap B = 10, applied before thinning)
 time: tau (days before conversion) = 7 U^1.4, sorted so the earliest touch comes first
 role: a touch at relative position u = (i+0.5)/n has P(upper) = 0.10 + 0.65 (1-u)^2,
       P(lower) = 0.10 + 0.65 u^2, P(mid) = rest
 campaign: within role by fixed popularity weights; with probability STICKY the touch repeats the previous campaign
 conversion: logit = B0 + sum_i THETA[c_i] 2^{-tau_i/7}; converting journeys are the accepted candidates
Composition calibration: 12 configurations of (THETA scale in {0.15, 0.25}, LENGTH_EXP in
{1.3, 1.6, 2.0}, STICKY in {0.3, 0.5}) were scanned for 45-55% of converting journeys with
>= 2 distinct campaigns and a meaningful share with >= 3; the first match in the prespecified
order was kept (THETA scale 0.15, B0 = -2.6, LENGTH_EXP = 1.6, STICKY = 0.3: about 53% / 37%).
Retention: p_j = s_j p_res + (1 - s_j) p_open with s_j in [0,1] containing both 0 and 1, so
max_j p_j / min_j p_j = p_open/p_res = r_true exactly. Each touch is retained independently
given its campaign, constant in time.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

K = 10
ROLE_OF = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])          # 0 upper, 1 mid, 2 lower
ROLE_NAMES = ["upper", "mid", "lower"]
CAP = 10
B_LINEAR = float(CAP)                                         # sum of <= 10 weights, each <= 1
LENGTH_EXP = 1.6
MAX_RAW_LEN = 40
STICKY = 0.30
TAU_POWER = 1.4
WINDOW = 7.0
H_DECAY = 7.0
ROLE_CAMPAIGN_WEIGHTS = {0: [0, 1, 2], 1: [3, 4, 5, 6], 2: [7, 8, 9]}
CAMPAIGN_POP = np.array([0.5, 0.3, 0.2, 0.35, 0.30, 0.20, 0.15, 0.40, 0.35, 0.25])
THETA = 0.15 * np.array([0.35 * m for m in (1.1, 1.0, 0.9)] + [0.55 * m for m in (1.2, 1.0, 0.9, 0.8)] + [0.90 * m for m in (1.1, 1.0, 0.9)])
B0 = -2.6
P_OPEN = 0.90
S_RES = np.array([1.0, 0.8, 0.55, 0.35, 0.2, 0.05, 0.0, 0.7, 0.4, 0.15])   # residual-platform share per campaign (contains 0 and 1)
P_LO_DECLARED_FACTOR = 1.0                                     # the declared floor equals the true minimum retention p_open / r
R_TRUE_GRID = (1.25, 1.5, 2.0, 2.5)


def retention_vector(r_true: float, p_open: float = P_OPEN) -> np.ndarray:
    """p_j = s_j p_res + (1 - s_j) p_open with p_res = p_open / r_true; max/min == r_true exactly because S_RES contains
    0 and 1.
    """
    if r_true < 1:
        raise ValueError("r_true >= 1")
    p_res = p_open / r_true
    p = S_RES * p_res + (1 - S_RES) * p_open
    return p


def check_retention(p: np.ndarray, r_true: float, p_lo: float, tol: float = 1e-12) -> dict:
    ok = {"in_(0,1]": bool(np.all(p > 0) and np.all(p <= 1)), "ratio": float(p.max() / p.min()),
          "ratio_matches": abs(p.max() / p.min() - r_true) <= tol * max(1.0, r_true), "floor_respected": bool(np.all(p >= p_lo - tol))}
    return ok


def sample_lengths(M: int, rng: np.random.Generator) -> np.ndarray:
    ks = np.arange(1, MAX_RAW_LEN + 1)
    pk = ks ** (-LENGTH_EXP); pk /= pk.sum()
    raw = rng.choice(ks, size=M, p=pk)
    return np.minimum(raw, CAP)           # keep the 10 most recent touches (cap on the LATENT journey)


def gen_paths(M: int, rng: np.random.Generator) -> dict:
    """M candidate latent paths before the conversion outcome is drawn, sorted by journey and, within a journey, earliest
    (largest tau) to latest.
    """
    n = sample_lengths(M, rng)
    total = int(n.sum())
    jid = np.repeat(np.arange(M), n)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    pos = np.arange(total) - np.repeat(start, n)
    nrep = np.repeat(n, n)
    u = (pos + 0.5) / nrep
    t_raw = WINDOW * rng.random(total) ** TAU_POWER
    order = np.lexsort((-t_raw, jid))
    tau = t_raw[order]
    p_up = 0.10 + 0.65 * (1 - u) ** 2; p_lo = 0.10 + 0.65 * u ** 2
    ur = rng.random(total)
    role = np.where(ur < p_up, 0, np.where(ur < p_up + (1 - p_up - p_lo), 1, 2))
    fresh = np.empty(total, dtype=np.int64)
    for rl, camps in ROLE_CAMPAIGN_WEIGHTS.items():
        sel = role == rl
        w = CAMPAIGN_POP[camps]; cw = np.cumsum(w / w.sum())
        fresh[sel] = np.array(camps)[np.searchsorted(cw, rng.random(int(sel.sum())), side="right").clip(0, len(camps) - 1)]
    repeat = (rng.random(total) < STICKY) & (pos > 0)
    base = np.where(~repeat, np.arange(total), 0)
    camp = fresh[np.maximum.accumulate(base)]
    return {"M": M, "n": n, "jid": jid, "pos": pos, "tau": tau, "camp": camp}


def conversion_logit(paths: dict) -> np.ndarray:
    w = np.bincount(paths["jid"], weights=THETA[paths["camp"]] * 2.0 ** (-paths["tau"] / H_DECAY), minlength=paths["M"])
    return B0 + w


def sample_converting(N: int, rng: np.random.Generator, batch: int = 400_000) -> tuple[dict, float]:
    """Draw candidate paths until N converting latent journeys are collected (acceptance order is arrival order).

    Returns (touch arrays re-indexed to 0..N-1, conversion rate of the candidate population).
    """
    got = 0; cand = 0; parts = []
    while got < N:
        pa = gen_paths(batch, rng)
        acc = rng.random(batch) < 1.0 / (1.0 + np.exp(-conversion_logit(pa)))
        keep_j = np.nonzero(acc)[0]
        if len(keep_j) > N - got:
            keep_j = keep_j[: N - got]
            cand += int(keep_j[-1]) + 1      # candidates actually consumed (the rest of the batch is discarded)
        else:
            cand += batch
        mask = np.zeros(batch, dtype=bool); mask[keep_j] = True
        tm = mask[pa["jid"]]
        newid = np.cumsum(mask) - 1
        parts.append({"jid": newid[pa["jid"][tm]] + got, "camp": pa["camp"][tm], "tau": pa["tau"][tm], "n": pa["n"][mask]})
        got += int(mask.sum())
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("jid", "camp", "tau", "n")}
    out["N"] = N
    return out, got / cand


def sample_warmup(M: int, rng: np.random.Generator) -> dict:
    """Independent warm-up dataset: M candidate paths with conversion labels (converting and not)."""
    pa = gen_paths(M, rng)
    y = rng.random(M) < 1.0 / (1.0 + np.exp(-conversion_logit(pa)))
    pa["y"] = y
    return pa


def thin(jid_camp: dict, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """In-model retention: each touch retained independently with probability p[campaign]."""
    return rng.random(len(jid_camp["camp"])) < p[jid_camp["camp"]]


class LinearSHAPModel:
    """Frozen logistic conversion model on per-campaign touch counts with exact interventional linear SHAP.

    For logit f(x) = b + beta.x and background mean mu, phi_j = beta_j (x_j - mu_j),
    identical to `shap.LinearExplainer` with feature_perturbation="interventional"
    (checked in tests). Credit is the positive SHAP on touched campaigns, normalised to
    sum 1; a conversion with no positive touched SHAP falls back to its last retained
    touch (as in models/logistic_shap.py). Fitted only on warm-up data (A1) and never refit.
    """

    def fit(self, warm: dict, C: float = 1.0, seed: int = 0):
        from sklearn.linear_model import LogisticRegression
        X = counts_matrix(warm["jid"], warm["camp"], None, warm["M"])
        y = warm["y"].astype(int)
        self.model = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(X, y)
        self.beta = self.model.coef_.ravel().copy()
        self.mu = X[y == 0].mean(0)        # background = warm-up NON-converting journeys (as models/logistic_shap.py)
        self.n_train = len(y)
        return self

    def credit(self, X: np.ndarray, last_onehot: np.ndarray, any_touch: np.ndarray) -> np.ndarray:
        phi = self.beta[None, :] * (X - self.mu[None, :])
        z = np.where(X > 0, np.clip(phi, 0.0, None), 0.0)
        s = z.sum(1, keepdims=True)
        cred = np.where(s > 0, z / np.where(s > 0, s, 1.0), last_onehot)
        return np.where(any_touch[:, None], cred, 0.0)


def counts_matrix(jid, camp, keep, N):
    w = np.ones(len(jid)) if keep is None else keep.astype(float)
    X = np.zeros((N, K))
    np.add.at(X, (jid, camp), w)
    return X


def last_touch_onehot(jid, camp, keep, N):
    idx = np.arange(len(jid)) if keep is None else np.nonzero(keep)[0]
    last = np.full(N, -1, dtype=np.int64)
    np.maximum.at(last, jid[idx], idx)          # arrays are ordered earliest -> latest within a journey
    oh = np.zeros((N, K)); has = last >= 0
    oh[np.nonzero(has)[0], camp[last[has]]] = 1.0
    return oh, has


def model_masses(model: str, trip: dict, keep, N: int, shap_model: LinearSHAPModel | None = None):
    """(Tj (N,K), B) for the given model on the touches of `trip` retained by `keep` (None = full latent).

    linear: retained-touch count. time_decay: weight 2^{-tau/7} belongs to the latent
    touch and is not renormalised after thinning. last_touch and logistic_shap are
    empirical (no fixed-weight theorem).
    """
    jid, camp, tau = trip["jid"], trip["camp"], trip["tau"]
    if model == "linear":
        return counts_matrix(jid, camp, keep, N), B_LINEAR
    if model == "time_decay":
        w = 2.0 ** (-tau / H_DECAY); w = w if keep is None else w * keep
        Tj = np.zeros((N, K)); np.add.at(Tj, (jid, camp), w)
        return Tj, B_LINEAR
    if model == "last_touch":
        oh, _ = last_touch_onehot(jid, camp, keep, N)
        return oh, 1.0
    if model == "logistic_shap":
        X = counts_matrix(jid, camp, keep, N)
        oh, has = last_touch_onehot(jid, camp, keep, N)
        return shap_model.credit(X, oh, has), 1.0
    raise ValueError(model)


MODELS = ("linear", "time_decay", "last_touch", "logistic_shap")
GUARANTEED = {"linear": "fixed-weight additive: theorem-covered (in-model)", "time_decay": "timestamp-fixed additive: theorem-covered for the aggregate share (in-model)",
              "last_touch": "EMPIRICAL: no fixed-weight identification theorem", "logistic_shap": "EMPIRICAL: learned model, no fixed-weight identification theorem"}


def pair_list():
    return list(itertools.combinations(range(K), 2))


def decision_outcomes(cert: dict, Delta: dict, delta: float):
    """cert[(j,k)] in {"j>k","k>j","unres"}; Delta[(j,k)] = A_j - A_k (full observation).

    A certification of j>k is incorrect if Delta_jk <= delta; a pair is actionable if
    |Delta| > delta. Returns per-pair arrays (|Delta|, actionable, correct, wrong, unres).
    """
    pl = pair_list()
    absD = np.array([abs(Delta[p]) for p in pl]); act = absD > delta
    wrong = np.zeros(len(pl), bool); correct = np.zeros(len(pl), bool); unres = np.zeros(len(pl), bool)
    for i, p in enumerate(pl):
        c, d = cert[p], Delta[p]
        if c == "unres":
            unres[i] = True
        elif c == "j>k":
            (correct if d > delta else wrong).__setitem__(i, True)
        else:
            (correct if d < -delta else wrong).__setitem__(i, True)
    return absD, act, correct, wrong, unres


LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0)


def loss_summary(absD, act, correct, wrong, unres):
    """Directional-error loss L_err = sum_actionable |Delta| 1{wrong}.

    Abstention-adjusted L_lambda = sum_actionable |Delta| (1{wrong} + lambda 1{UNRES});
    both are also normalised by sum_actionable |Delta|. Resolved-opportunity fraction =
    (# actionable pairs certified correctly) / (# actionable pairs). An attribution-ranking
    decision loss, not causal budget-allocation regret.
    """
    a = act
    tot = float(absD[a].sum())
    out = {"n_actionable": int(a.sum()), "sum_abs_delta_actionable": tot,
           "resolved_opportunity_fraction": float((correct & a).sum() / max(a.sum(), 1)),
           "wrong_actionable": int((wrong & a).sum()), "wrong_nonactionable": int((wrong & ~a).sum()),
           "unresolved_actionable": int((unres & a).sum()),
           "directional_error_loss": float((absD * (wrong & a)).sum()),
           "directional_error_loss_normalised": float((absD * (wrong & a)).sum() / tot) if tot > 0 else 0.0}
    for lam in LAMBDAS:
        L = float((absD * ((wrong & a) + lam * (unres & a))).sum())
        out[f"abstention_adjusted_loss_lambda={lam}"] = L
        out[f"abstention_adjusted_loss_normalised_lambda={lam}"] = L / tot if tot > 0 else 0.0
    return out
