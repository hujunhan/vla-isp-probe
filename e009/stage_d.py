"""E009 Stage D (notebook E009 "Stage D", frozen 2026-10-13): RAW (noise4) vs 8-bit (u8_noise4) on Goal 0 and
Object 0, init states 20-39, noise seeds 0 (results/e009) + 1, 2 (results/e009_s1, _s2).
Unit = (task, init state); per unit and arm, success averaged over the 3 seeds; d = u8_noise4 - noise4.
Primary: mean d (points), two-sided sign-flip permutation test over units (100k), bootstrap 95% CI over units.
Rule: harsher if mean d >= 10 and p < 0.05; no resolvable difference if p >= 0.05; else harsher by < 10.
[ai-scaffold]"""
import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import fisher_exact

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
TASKS = {("libero_goal", "0"), ("libero_object", "0")}
ARMS = ("noise4", "u8_noise4")


def main():
    succ = defaultdict(list)          # (arm, suite, task, init) -> [success per seed]
    for seed, d in enumerate(("e009", "e009_s1", "e009_s2")):
        for r in csv.DictReader(open(os.path.join(ROOT, d, "oft.csv"))):
            if r["condition"] in ARMS and (r["suite"], r["task_id"]) in TASKS and 20 <= int(r["episode"]) < 40:
                succ[(r["condition"], r["suite"], r["task_id"], int(r["episode"]))].append(int(r["success"]))
    units = sorted({k[1:] for k in succ})
    n_seeds = {a: [len(succ[(a, *u)]) for u in units] for a in ARMS}
    assert len(units) == 40 and all(n == 3 for a in ARMS for n in n_seeds[a]), n_seeds
    m = {a: np.array([np.mean(succ[(a, *u)]) for u in units]) for a in ARMS}
    d = 100 * (m["u8_noise4"] - m["noise4"])
    obs = d.mean()
    rng = np.random.default_rng(0)
    flips = rng.choice([-1, 1], size=(100_000, len(d)))
    p = float((np.abs((flips * d).mean(1)) >= abs(obs) - 1e-12).mean())
    boot = d[rng.integers(0, len(d), size=(100_000, len(d)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("per arm, per task (successes / 60):")
    for a in ARMS:
        for t in sorted(TASKS):
            k = sum(sum(succ[(a, *u)]) for u in units if u[:2] == t)
            print(f"  {a:10s} {t[0]}:{t[1]}  {k}/60 = {100 * k / 60:.1f}%")
    print("per seed (both tasks, successes / 40):")
    for a in ARMS:
        print(f"  {a:10s}", [sum(succ[(a, *u)][s] for u in units) for s in range(3)])
    ka, kb = (int(round(m[a].sum() * 3)) for a in ARMS)
    print(f"pooled: noise4 {ka}/120, u8_noise4 {kb}/120, Fisher two-sided p = "
          f"{fisher_exact([[ka, 120 - ka], [kb, 120 - kb]])[1]:.3g} (not decisive)")
    print(f"mean d = {obs:+.1f} points, 95% CI [{lo:+.1f}, {hi:+.1f}], permutation p = {p:.4g}")
    # units where arms disagree; seed-to-seed variability
    var = {a: float(np.mean([0 < m[a][i] < 1 for i in range(len(units))])) for a in ARMS}
    print("share of init states with mixed outcomes across seeds:", var)
    if p >= 0.05:
        v = "NO RESOLVABLE DIFFERENCE"
    elif obs >= 10:
        v = "RAW PATH HARSHER on the fragile tasks"
    else:
        v = "harsher, but by < 10 points" if obs > 0 else "8-bit path harsher"
    print("Rule:", v)


if __name__ == "__main__":
    main()
