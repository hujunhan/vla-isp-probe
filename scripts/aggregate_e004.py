"""Aggregate E004 (camera/ISP probe) per-episode CSVs and apply stopping-rule tests S3 and S4.

Definitions (decisions/003, E004 plan):
  * rate = successes / valid episodes (episodes with an "Episode error" excluded and counted apart);
    pooled over the 3 tasks (150 episodes per condition per policy).
  * 95% CI: Wilson.
  * Reference for every camera condition: clean_cam (same RAW round trip, no noise). render vs
    clean_cam is Block 0b.
  * S3 (effect exists): a condition counts if drop vs clean_cam >= 10 points AND the two Wilson CIs
    do not overlap. S3 passes only if, for BOTH policies, some PLAUSIBLE condition counts.
    The pre-registered plan defines plausible RANGES for knob A only (shot noise in [1e-4, 1.2e-2],
    Brooks #30 Sec. 3.1; over-exposure +1 EV). Whether knob-B variants (denoise off/strong) at those
    noise levels count as "plausible settings" was NOT specified, so S3 is reported two ways:
      strict    = knob-A conditions only (default denoise)
      inclusive = knob-A conditions + knob-B variants at plausible noise
    The owner decides which reading applies. stress_* conditions are reported, never used.
  * S4 (novel angle): RAW condition vs image-space condition at matched PSNR; S4 fails only if
    |difference| < 5 points for every matched pair (all levels, both policies).

Usage: python aggregate_e004.py results/e004/oft.csv results/e004/oft_w.csv

[ai-scaffold] — advisor-written metric code; owner explain-back owed (decisions/006).
"""
import csv
import math
import sys
from collections import defaultdict

MATCHED = {"lam1e-3": "img_sigma_lam1e-3", "lam5e-3": "img_sigma_lam5e-3",
           "lam1.2e-2": "img_sigma_lam1.2e-2", "stress_lam3e-2": "img_sigma_lam3e-2"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def plausible(tag):
    return not tag.startswith(("stress", "img_", "render", "clean_cam", "linear", "ag_only"))


def strict(tag):
    return plausible(tag) and "_dn" not in tag


def main(paths):
    cells = defaultdict(lambda: [0, 0, 0])            # (policy, condition, suite|ALL) -> n, k, err
    for p in paths:
        policy = p.rsplit("/", 1)[-1].replace(".csv", "")
        for r in csv.DictReader(open(p)):
            for s in (r["suite"], "ALL"):
                c = cells[(policy, r["condition"], s)]
                if int(r["error"]):
                    c[2] += 1
                else:
                    c[0] += 1
                    c[1] += int(r["success"])

    def rate(policy, cond, suite="ALL"):
        n, k, _ = cells.get((policy, cond, suite), (0, 0, 0))
        return (100 * k / n if n else float("nan")), n, k

    policies = sorted({k[0] for k in cells})
    s3_pass = {}
    for pol in policies:
        conds = sorted({k[1] for k in cells if k[0] == pol}, key=lambda t: (not t.startswith(("render", "clean")), t))
        ref, ref_n, ref_k = rate(pol, "clean_cam")
        ref_lo, ref_hi = wilson(ref_k, ref_n)
        print(f"\n## {pol}  (reference clean_cam = {ref:.1f}% [{ref_lo:.1f}, {ref_hi:.1f}], n={ref_n})\n")
        print("| condition | n | errors | success % [95% CI] | drop vs clean | S3 counts? | spatial | object | goal |")
        print("|---|---|---|---|---|---|---|---|---|")
        hits = []
        for cond in conds:
            r, n, k = rate(pol, cond)
            err = cells[(pol, cond, "ALL")][2]
            lo, hi = wilson(k, n)
            drop = ref - r
            counts = drop >= 10 and hi < ref_lo
            flag = "—" if not plausible(cond) else (("**yes**" if counts else "no") + ("" if strict(cond) else " (knob B)"))
            if plausible(cond) and counts:
                hits.append(cond)
            per = " | ".join(f"{rate(pol, cond, s)[2]}/{rate(pol, cond, s)[1]}"
                             for s in ("libero_spatial", "libero_object", "libero_goal"))
            print(f"| {cond} | {n} | {err} | {r:.1f} [{lo:.1f}, {hi:.1f}] | {drop:+.1f} | {flag} | {per} |")
        s3_pass[pol] = hits
        print(f"\nS3 plausible conditions with a qualifying drop: {hits or 'none'}")

        print("\nS4 matched-PSNR pairs (RAW vs image-space):\n")
        print("| level | RAW % | image-space % | RAW − image | |diff| ≥ 5? |")
        print("|---|---|---|---|---|")
        for raw_tag, img_tag in MATCHED.items():
            a, na, _ = rate(pol, raw_tag)
            b, nb, _ = rate(pol, img_tag)
            if na and nb:
                print(f"| {raw_tag} | {a:.1f} | {b:.1f} | {a - b:+.1f} | {'yes' if abs(a - b) >= 5 else 'no'} |")

    import json, os
    plan = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e004", "conditions.json")))
    complete = True
    for pol, blocks in plan["policies"].items():
        for b in blocks:
            for c in plan["blocks"][b]:
                if rate(pol, c["tag"])[1] + cells[(pol, c["tag"], "ALL")][2] < 150:
                    complete = False
    for name, sel in (("strict", strict), ("inclusive", plausible)):
        ok = all(any(sel(c) for c in s3_pass.get(p, [])) for p in ("oft", "oft_w"))
        verdict = ("PASS" if ok else "FAIL") if complete else ("interim: " + ("PASS so far" if ok else "no pass yet"))
        print(f"**S3 {name}: {verdict}**")


if __name__ == "__main__":
    main(sys.argv[1:])
