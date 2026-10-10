"""Aggregate E005 (R1) and apply the pre-registered replication rule (E005 record).

For each policy and each noise condition, compare with its reference condition (results/e005/
conditions.json "references"):
  (1) pooled drop >= 10 points with non-overlapping Wilson 95% CIs (n = 180 per condition), and
  (2) point-estimate drop >= 10 points in >= 2 of the 3 suites (n = 60 per suite).
R1 replicates iff, for BOTH policies, at least one Infinite-ISP condition (tag inf_lam*) meets (1)
and (2). Toy conditions are reported as the secondary arm under the same rule.
Episodes with an "Episode error" are excluded from rates and counted separately.

Usage: python aggregate_e005.py results/e005/oft.csv results/e005/oft_w.csv

[ai-scaffold] — advisor-written metric code; owner explain-back owed.
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

SUITES = ("libero_spatial", "libero_object", "libero_goal")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main(paths):
    exp = os.environ.get("EXP", "e005")          # EXP=e006 for the fresh-episode confirmation
    plan = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", exp,
                                       "conditions.json")))
    refs = plan["references"]
    expected_n = plan.get("episodes_per_task", 20) * 9
    cells = defaultdict(lambda: [0, 0, 0])   # (policy, cond, suite|ALL) -> n_valid, k, errors
    for p in paths:
        pol = os.path.basename(p).replace(".csv", "")
        for r in csv.DictReader(open(p)):
            for s in (r["suite"], "ALL"):
                c = cells[(pol, r["condition"], s)]
                if int(r["error"]):
                    c[2] += 1
                else:
                    c[0] += 1
                    c[1] += int(r["success"])

    def rate(pol, cond, s="ALL"):
        n, k, _ = cells.get((pol, cond, s), (0, 0, 0))
        return (100.0 * k / n if n else float("nan")), n, k

    verdict = {}
    for pol in sorted({k[0] for k in cells}):
        print(f"\n## {pol}\n")
        print("| condition | n | success % [95% CI] | reference | ref % | drop | CI separate | suites ≥10 drop | qualifies |")
        print("|---|---|---|---|---|---|---|---|---|")
        conds = sorted({k[1] for k in cells if k[0] == pol})
        for cond in [c for c in conds if c not in refs]:
            r, n, k = rate(pol, cond)
            lo, hi = wilson(k, n)
            print(f"| {cond} | {n} | {r:.1f} [{lo:.1f}, {hi:.1f}] | — | — | — | — | — | — |")
        hits = []
        for cond in [c for c in conds if c in refs]:
            ref = refs[cond]
            r, n, k = rate(pol, cond)
            rr, rn, rk = rate(pol, ref)
            lo, hi = wilson(k, n)
            rlo, rhi = wilson(rk, rn)
            drop = rr - r
            sep = hi < rlo
            suite_drops = [rate(pol, ref, s)[0] - rate(pol, cond, s)[0] for s in SUITES]
            n_suites = sum(d >= 10 for d in suite_drops)
            complete = n + cells[(pol, cond, "ALL")][2] >= expected_n
            q = drop >= 10 and sep and n_suites >= 2
            if q and complete:
                hits.append(cond)
            sd = ", ".join(f"{d:+.0f}" for d in suite_drops)
            status = ("**yes**" if q else "no") + ("" if complete else " (partial)")
            print(f"| {cond} | {n} | {r:.1f} [{lo:.1f}, {hi:.1f}] | {ref} | {rr:.1f} | {drop:+.1f} | {sep} | "
                  f"{n_suites}/3 ({sd}) | {status} |")
        verdict[pol] = hits
    inf_ok = all(any(h.startswith("inf_") for h in verdict.get(p, [])) for p in ("oft", "oft_w"))
    toy_ok = all(any(h.startswith("toy_") for h in verdict.get(p, [])) for p in ("oft", "oft_w"))
    print(f"\n**R1 (primary, Infinite-ISP): {'REPLICATES' if inf_ok else 'FAILS'}** — qualifying: "
          f"{ {p: [h for h in v if h.startswith('inf_')] for p, v in verdict.items()} }")
    print(f"Secondary (toy ISP on 9 tasks): {'replicates' if toy_ok else 'does not replicate (or incomplete)'} — "
          f"{ {p: [h for h in v if h.startswith('toy_')] for p, v in verdict.items()} }")


if __name__ == "__main__":
    main(sys.argv[1:])
