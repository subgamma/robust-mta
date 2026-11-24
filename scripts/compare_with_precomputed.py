"""Compare freshly generated results with the shipped summaries.

    python scripts/compare_with_precomputed.py                 # results/ against precomputed_results/, tolerance 0
    python scripts/compare_with_precomputed.py --tol 1e-9 results/e9

Every JSON file under precomputed_results/ that also exists in the results directory is compared leaf by
leaf; `wall_time*` fields are ignored and CSV files are compared as text. Reference files missing from the
results directory are listed and do not count as differences. Exit status is 1 if any value differs by more
than the tolerance.
"""
import argparse
import json
import os
import sys


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if str(k).startswith("wall_time"):
                continue
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def differs(a, b, tol):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) > tol
    return a != b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", default="results")
    ap.add_argument("--reference", default="precomputed_results")
    ap.add_argument("--tol", type=float, default=0.0, help="absolute tolerance for numeric leaves (default 0)")
    ap.add_argument("only", nargs="*", help="restrict to files below these paths (relative to --fresh)")
    a = ap.parse_args()
    n_files = n_bad = n_missing = 0
    for dp, _, fs in sorted(os.walk(a.reference)):
        for f in sorted(fs):
            rel = os.path.relpath(os.path.join(dp, f), a.reference)
            if a.only and not any(os.path.join(a.fresh, rel).startswith(o.rstrip("/")) for o in a.only):
                continue
            fresh = os.path.join(a.fresh, rel)
            if not os.path.exists(fresh):
                n_missing += 1
                continue
            if f.endswith(".json"):
                ref = dict(leaves(json.load(open(os.path.join(dp, f)))))
                new = dict(leaves(json.load(open(fresh))))
                bad = [k for k in set(ref) | set(new) if k not in ref or k not in new or differs(ref[k], new[k], a.tol)]
            elif f.endswith(".csv"):
                bad = [] if open(os.path.join(dp, f)).read() == open(fresh).read() else ["(file text)"]
            else:
                continue
            n_files += 1
            if bad:
                n_bad += 1
                print(f"DIFFERS  {rel}: {len(bad)} values, e.g. {sorted(bad)[:2]}")
    print(f"compared {n_files} files, {n_bad} differ (tolerance {a.tol}); {n_missing} reference files not present in {a.fresh}/")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
