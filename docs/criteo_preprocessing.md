# Criteo preprocessing

Source: `criteo/criteo-attribution-dataset` on Hugging Face
(https://huggingface.co/datasets/criteo/criteo-attribution-dataset),
`criteo_attribution_dataset.tsv.gz`, CC-BY-NC-SA-4.0. Size (653,015,824 bytes)
and SHA-256 are verified by `scripts/download_criteo.py`. Stored at
`data/criteo_attribution_dataset.tsv.gz` (git-ignored; not redistributed).

## Schema verification

Downloaded file's header matches the expected columns from the dataset card
exactly: `timestamp, uid, campaign, conversion, conversion_timestamp,
conversion_id, attribution, click, click_pos, click_nb, cost, cpo,
time_since_last_click, cat1..cat9`. Verified programmatically in
`anytime_attribution.criteo.load_raw` (raises on any missing column) and in
`tests/test_criteo.py::test_load_raw_rejects_bad_schema`.

## Journey reconstruction: verified assumptions

- **Every `conversion=1` row also has `click=1`.** Checked exhaustively
  (not sampled) over all 16,468,027 rows: 806,196 rows have `conversion=1`;
  0 of those have `click=0`. So "touches on a converting journey" and
  "conversion=1 rows" are the same set — no separate click-filtering step
  needed for journey reconstruction (`anytime_attribution.criteo.converting_touches`).
- **`conversion_id` is NOT globally unique across users.** 2,910 of 435,810
  distinct `conversion_id` values are shared by more than one `uid`.
  keying journeys on `(uid, conversion_id)` rather than `conversion_id` alone is therefore required:
  grouping by `conversion_id` alone would silently merge 2,910 pairs of
  distinct converting journeys belonging to different users into one.
  `anytime_attribution.criteo.journey_key` uses the pair; regression-guarded
  by `tests/test_criteo.py::test_journey_key_disambiguates_colliding_conversion_ids`.
- **`timestamp` and `conversion_timestamp` are in raw seconds**, range
  `[0, 2,671,199]` for `timestamp` (≈30.9 days — the impression window) and
  up to `5,262,888` for `conversion_timestamp` (≈60.9 days — conversions are
  tracked for up to ~30 days after the last impressions, so late touches can
  convert well past the nominal 30-day monitoring window). "Day" is defined
  as `conversion_timestamp // 86400 + 1` (1-indexed) throughout.

## Filter-step counts (funnel)

| Step | Rows / journeys | Notes |
|---|---|---|
| Raw rows | 16,468,027 | Full downloaded file |
| Rows with `conversion=1` (= touches on converting journeys) | 806,196 | All verified `click=1` |
| Distinct converting journeys, keyed `(uid, conversion_id)` | 438,730 | vs. 435,810 if keyed on `conversion_id` alone (the 2,910-journey undercount from the collision above) |
| — warm-up (`conv_day` 1–7) | 65,372 | Used to fix scope (cap, top-K campaigns) and to train models (A1) |
| — monitored (`conv_day` 8–30) | 274,337 | Credits materialized here; certification runs here |
| — after day 30 (discarded) | 99,021 | Touches near the end of the ~31-day impression window converting up to ~30 days later; out of scope for this study, not read further |
| Restricted to top-20 campaigns (warm-up) | 33,324 journeys / 43,915 touch-rows | Campaigns fixed from warm-up only, see below |
| Restricted to top-20 campaigns (monitored) | 123,439 journeys (45.0% of the 274,337 monitored conversion events) / 208,101 touch-rows | Same fixed campaign set applied |
| After A4 cap, warm-up | 33,324 journeys / 43,648 touch-rows | 267 touch-rows (0.6%) dropped by the 7-day lookback; 0 journeys dropped entirely |
| After A4 cap, monitored | 109,221 journeys (39.8% of the 274,337 monitored conversion events) / 160,378 touch-rows; **14,218 top-20 journeys (11.5% of the 123,439) are dropped entirely**, because every recorded touch precedes the 7-day lookback (median lag 11.4 days, maximum 29) | 47,723 touch-rows (22.9%) dropped by the 7-day lookback (monitored journeys can have touches up to ~29 days old, unlike warm-up journeys which structurally cannot exceed a ~7-day lookback); max touches per journey after cap: 10 in both windows |

**Population restriction (checked against the raw file).** In the warm-up window no
journey is dropped entirely by the lookback (33,324 of 33,324 retained). In the
monitored window this is **not** true: of the 123,439 monitored conversions on a
top-20 campaign, 14,218 (11.5%) have every recorded touch older than the 7-day
lookback and therefore carry zero capped mass; they are outside the estimand
(Assumption A4: the cap is part of the estimand) and are not in the 109,221-row
primary stream. The processed stream is thus 109,221 of the 274,337 monitored
conversion events (39.8%). The 7-day warm-up cannot observe lags above seven days
(left-censoring by construction), so "essentially all mass lies within 7 days"
does not carry over to the monitored window, where 22.9% of top-20 touch rows lie
beyond 7 days. The filter (tracked campaign AND at least one touch within the
lookback) depends on latent journey properties, not on retention. Only the
max-touch cap never dropped a journey entirely (every retained journey kept at
least one touch). Additionally, 14,644 of the 213,920 monitored users (6.8%) also
have a warm-up conversion; no touch row is shared between a warm-up and a
monitored conversion.

Journeys with **zero** touches remaining after the top-20 campaign restriction
(every touch on a non-top-20 campaign) are outside this study's scope: their
conversions are real but unattributable within a fixed K=20 framework. Exact
counts: `65,372 − 33,324 = 32,048` (warm-up) and `274,337 − 123,439 = 150,898`
(monitored), plus the 14,218 lookback exclusions above.

## A4 cap: chosen from warm-up (days 1–7) only

The lookback window and max-touch rule are fixed from warm-up statistics only,
before the monitored stream is read for anything beyond computing these two numbers.

**Journey length (touch count), warm-up, before any campaign restriction:**

| Quantile | 50% | 75% | 90% | 95% | 99% | 99.9% | 100% (max) |
|---|---|---|---|---|---|---|---|
| n_touches | 1 | 1 | 2 | 3 | 6 | 15 | 40 |

**Touch-to-conversion lag `tau` in days, warm-up:**

| Quantile | 50% | 75% | 90% | 95% | 99% | 99.9% | 100% (max) |
|---|---|---|---|---|---|---|---|
| tau_days | 0.044 | 1.159 | 3.037 | 4.072 | 5.634 | 6.284 | 6.940 |

The warm-up window itself mechanically bounds `tau` for early-converting
journeys (a day-1 conversion cannot have a touch more than ~1 day old), so
this distribution is somewhat left-censored. Restricting to conversions on
day 7 exactly (the only day within warm-up with a full week of preceding
history available) gives an uncensored read: 99th pct = 6.19 days, 99.9th =
6.47, max = 6.94 — essentially all mass is within 7 days even with the full
week available.

**Chosen cap** (`config/criteo.yaml`):
- `lookback_days = 7` — a round number matching the warm-up window itself;
  the day-7-only check above shows negligible mass beyond it *within the warm-up window* (which cannot, by construction, contain lags above seven days; see the population-restriction paragraph above for the monitored window).
- `max_touches = 10` — covers 99.73% of warm-up journeys uncapped (between
  the 99th pct of 6 and 99.9th pct of 15):

| max_touches | 5 | 6 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|
| % warm-up journeys uncapped | 98.603 | 99.093 | 99.533 | 99.732 | 99.917 | 99.963 |

## Top-K campaigns: chosen from warm-up (days 1–7) only

Ranked by distinct converting-journey count in warm-up only (663 distinct
campaigns are active in warm-up out of 675 in the full dataset). K=20 is the
primary ((uid, conversion_id)) construction's scope. There is no separate K=10 run on the primary
construction; K=10 is used only on the secondary (literature-convention
linkage) construction below, ranked by a *different* criterion (exposure,
not conversions).

Top-20 campaign IDs (warm-up journey counts, descending):
`32368244` (3717), `15184511` (3550), `9100693` (3403), `5544859` (2594),
`2869134` (2569), `10341182`, `9100689`, `16184517`, `9100692`, `30801593`,
`28351001`, `17710664`, `9100690`, `9100691`, `15398570`, `26891650`,
`5061834`, `16922675`, `29427842`, `14121532` (down to 497 journeys for the
20th).

## Config hash

`config/criteo.yaml` content hash (`anytime_attribution.criteo.config_hash`): **`45ea48be0245c6bf`**.
The primary-relevant fields (`cap`, `campaigns.top_20`, `days`, `certification`) are unchanged by the
`secondary:` section, so the primary credits (`results/credits/`) and the secondary credits
(`results/credits_secondary/`) are both derived from this configuration.

## Non-converting pool (for model training only)

Rows with `conversion=0`: 15,661,831 (16,468,027 − 806,196), of which
5,141,367 have `click=1`. These are used only as negative
examples for training the logistic and DARNN conversion-prediction models
(`models/logistic_shap.py`, `models/darnn.py`), never as "journeys" in the
attribution sense and never as part of any credit-materialization or
certification computation. Negative-sampling parameters are in `scripts/materialize_criteo_credits.py` and `anytime_attribution/criteo.py`.

## Literature-convention linkage (secondary construction)

The primary
`(uid, conversion_id)` construction is single-campaign by construction
(every journey touches exactly one campaign, as verified above),
so within-journey mechanisms and model-to-model credit differences have no real data to exercise. This
section reproduces the construction used by CAMTA and CausalMTA on this
same dataset: cross-campaign journeys arise by **time linkage** (a user's
touches within a lookback window of a conversion), not by the conversion
identifier. **Every artifact from this construction is labelled
"literature-convention linkage"** — it is not the primary construction and
carries its own linkage assumption, stated below.

### Choices the source papers do not specify

CAMTA and CausalMTA's published descriptions of their Criteo preprocessing
do not fully pin down an exact, re-runnable procedure. Every choice made to
fill that gap is listed here (also in `criteo.py`'s module docstring):

1. **"Most exposed"** is ranked by raw impression count (every row, not
   just clicks or conversions) in warm-up — "exposure" read literally, not
   click-through or conversion volume. This is a genuinely different
   ranking from the primary's top-20 (by conversions): 7 of the top-10
   exposure campaigns also appear in the primary's top-20 list; 3
   (`17686799`, `30491418`, `31772643`) are high-impression-volume
   campaigns that do not make the primary's conversion-based list at all.
2. **The lookback window is the same 7 days** fixed for the primary
   construction (Assumption A4) — not a separately-tuned value, since
   neither paper states one for this dataset and reusing the already-fixed
   A4 window avoids introducing an extra free parameter.
3. **A `max_touches=10` cap is still applied** after windowing, for
   consistency with Assumption A4 (a stated bound on journey mass `S`,
   needed throughout the identification framework); the source papers do
   not state an upper cap. In practice this rarely binds — see funnel below.
4. **Linkage scope**: a touch is included in a conversion's journey if it
   is on a top-K campaign, by the same user, within the lookback window —
   **regardless of which `conversion_id` (if any) it was originally
   recorded against.** This is the linkage assumption of the construction: a touchpoint on campaign k
   preceding a conversion on campaign j is treated as part of j's journey because of proximity in time, not
   because of any recorded association. Results under it are conditional on it, not a claim that the two
   touches are causally or operationally related.
5. **The `>3`-touchpoint filter is applied to both classes.** Applying it
   only to positive (converting) journeys, while leaving negative
   (non-converting) pseudo-journeys unfiltered (which mostly have 1–2
   touches — see the funnel below), would let the conversion classifier
   trivially learn "more touches predicts conversion" from the filter
   itself rather than from campaign-level signal. Applying the same filter
   to both classes keeps the touch-count distributions comparable.
6. **Negative pseudo-journeys are anchored at the user's last top-K
   impression** in the window (not a real conversion time), matching the
   anchor convention already used for the primary construction's negatives
   (`build_negative_examples`).

### Top-10 campaigns by warm-up exposure (impression count)

`30801593` (143,674), `10341182` (105,214), `17686799` (89,209), `5061834`
(84,688), `15398570` (71,189), `30491418` (62,149), `15184511` (61,036),
`28351001` (56,542), `31772643` (51,369), `32368244` (46,167).

### Filter definition

The implemented filter keeps journeys with **more than three** touchpoints
(`len(touches) > min_touches` with `min_touches = 3`, i.e. a minimum of four;
the observed minimum is 4, and 20,089 monitored journeys survive versus 27,495 with at
least three). The windowing uses `timestamp <= anchor` (non-strict), so a touch with exactly the same
second-resolution timestamp as the anchor can enter a journey; strict-before ordering is not enforced.

### Funnel counts

| Step | Warm-up (days 1–7) | Monitored (days 8–30) |
|---|---|---|
| Conversion events whose own campaign is in the top-10 exposure set | 15,166 (12,336 distinct users) | — (materialized directly; candidate count not separately logged) |
| After time-linkage windowing (all top-10 touches, any original conversion_id, within 7-day lookback) + `max_touches=10` cap | 15,166 (cap rarely binds; see below) | — |
| After the `>3`-touchpoint filter (final positive population) | **3,690** | **20,089** |
| — of which multi-campaign (≥2 distinct campaigns touched) | 117 (3.2%) | 804 (4.0%) |
| Candidate non-converting users (active on a top-10 campaign, warm-up) | 444,645 | — (negatives are warm-up only, used for training) |
| Sampled candidates (before the touchpoint filter) | 200,000 | — |
| Negative pseudo-journeys surviving the `>3`-touchpoint filter | **15,139** (7.57% survival rate) | — |

Pre-filter touch-count distribution matches the primary's warm-up shape
(median 1–2 touches; most conversion events have few touches before
linkage), which is exactly why the `>3`-touchpoint filter removes roughly
three-quarters of candidates (15,166 → 3,690): CAMTA/CausalMTA's convention
selects for the richer-history subpopulation where within-journey
mechanisms are actually exercisable, at the cost of population size and
representativeness.

### Model credit agreement

On the monitored secondary population, `last_touch`/`linear`/`time_decay`
agree exactly on 95.998% of the 20,089 journeys (the single-campaign
majority) and **disagree on the remaining 4.0%** — matching the
multi-campaign fraction exactly, confirming the construction does what it
is meant to do. Example (10-touch journey spanning 2 campaigns):
`last_touch=[…,1,…]` (all credit to the last-touched campaign),
`linear=[…,0.10,…,0.90]`, `time_decay=[…,0.118,…,0.882]`,
`logistic_shap=[…,0,…,1]`, `darnn=[…,0.113,…,0.887]` — four distinct
credit splits from the same journey.

## Dataset-card reconciliation

The Hugging Face dataset card states "45K conversions." This does not match
either this project's or the literature's conversion-linked journey counts:
this project finds **438,730** distinct `(uid, conversion_id)` journeys
(all `conv_day` 1–61; see the primary-construction funnel above), and
CausalMTA reports **435,810** converting journeys on the same released file
— the two are close (438,730 vs 435,810, a 0.7% difference plausibly from a
different raw-file snapshot or minor preprocessing variant) and both are
roughly **10x** the card's stated 45K. The most likely explanation is that
the card's "45K" refers to a different unit than "conversion-linked
journeys" — e.g. distinct `conversion_id` values before deduplicating
across the 2,910 colliding `(uid, conversion_id)` pairs would move the
count in the wrong direction (down by only ~3K, not by a factor of ~10), so
this alone does not explain the gap; a more likely candidate is that "45K"
describes something else entirely (e.g. a distinct campaign-conversion
pair count at a coarser granularity, or a stale/superseded card figure) —
**this is unresolved**. The measured value is ~4.4×10^5 conversion-linked journeys, matching
CausalMTA; the card's figure is treated as unreconciled and no cause is asserted.
