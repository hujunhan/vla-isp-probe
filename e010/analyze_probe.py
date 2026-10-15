"""E010 analysis (notebook experiments/E010-policy-specific-fragility.md, frozen 2026-10-13).
Per query: d_trans = mean over N variants x 8 chunk steps of |a_noisy - a_clean| on dims 0-2, / pooled std of clean
translation actions (all queries, both policies); d_rot likewise for dims 3-5; gripper flip rate = share of sign
disagreements on dim 6. Per (policy, task): S = mean d_trans, S90 = p90; same for d_rot, flip.
Closed-loop failure per cell: OFT noise4 (results/e009/oft.csv), OFT_w t4 (results/e009/oft_w.csv), seed 0.
Rule: H1 supported if Spearman(S, failure) >= 0.5 with p < 0.05 AND >= 3/4 cross comparisons; not supported if
rho < 0.3 or <= 2/4; else partial. [ai-scaffold]"""
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PROBE = os.path.join(ROOT, os.environ.get("E010_DIR", "e010"), "probe")
FAIL_SRC = {"oft": ("e009/oft.csv", "noise4"), "oft_w": ("e009/oft_w.csv", "t4")}
CROSS = [(("libero_goal", "0"), "oft", "oft_w"), (("libero_object", "0"), "oft", "oft_w"),
         (("libero_spatial", "1"), "oft_w", "oft"), (("libero_object", "1"), "oft_w", "oft")]


def load():
    eps = []
    for f in sorted(glob.glob(os.path.join(PROBE, "*.npz"))):
        m = re.match(r"(oft_w|oft)_(libero_\w+?)_(\d+)_(\d+)\.npz", os.path.basename(f))
        z = np.load(f)
        eps.append(dict(policy=m[1], suite=m[2], task=m[3], ep=int(m[4]), step=z["step"], clean=z["clean"],
                        noisy=z["noisy"], repeat=z["clean_repeat"], success=int(z["success"])))
    return eps


def failure_rates():
    out = {}
    for pol, (f, tag) in FAIL_SRC.items():
        k = defaultdict(lambda: [0, 0])
        for r in csv.DictReader(open(os.path.join(ROOT, f))):
            if r["condition"] == tag:
                k[(r["suite"], r["task_id"])][0] += 1 - int(r["success"])
                k[(r["suite"], r["task_id"])][1] += 1
        out.update({(pol, *t): v[0] / v[1] for t, v in k.items()})
    return out


def main():
    eps = load()
    det = max(float(np.abs(e["repeat"] - e["clean"][0]).max()) for e in eps)
    allc = np.concatenate([e["clean"].reshape(-1, 7) for e in eps])
    sd_t, sd_r = allc[:, :3].std(), allc[:, 3:6].std()
    cell = defaultdict(lambda: {"t": [], "r": [], "g": [], "near_grasp": []})
    for e in eps:
        c, n = e["clean"][:, None], e["noisy"]                          # (Q,1,8,7), (Q,N,8,7)
        dt = np.abs(n[..., :3] - c[..., :3]).mean((1, 2, 3)) / sd_t
        dr = np.abs(n[..., 3:6] - c[..., 3:6]).mean((1, 2, 3)) / sd_r
        g = (np.sign(n[..., 6]) != np.sign(c[..., 6])).mean((1, 2))
        # "near grasp": queries whose clean chunk changes gripper sign, or the query before/after one
        ch = np.array([len(set(np.sign(q[:, 6]))) > 1 for q in e["clean"]])
        if len(e["clean"]) > 1:
            prev = np.sign(e["clean"][:-1, -1, 6]) != np.sign(e["clean"][1:, 0, 6])
            ch[1:] |= prev
        near = ch | np.roll(ch, 1) | np.roll(ch, -1)
        k = cell[(e["policy"], e["suite"], e["task"])]
        k["t"] += dt.tolist(); k["r"] += dr.tolist(); k["g"] += g.tolist(); k["near_grasp"] += near.tolist()
    fail = failure_rates()
    rows = []
    for key in sorted(cell):
        v = {m: np.array(cell[key][m]) for m in ("t", "r", "g")}
        ng = np.array(cell[key]["near_grasp"], bool)
        rows.append(dict(policy=key[0], suite=key[1], task=key[2], n_queries=len(v["t"]),
                         S=float(v["t"].mean()), S90=float(np.percentile(v["t"], 90)),
                         R=float(v["r"].mean()), G=float(v["g"].mean()),
                         S_grasp=float(v["t"][ng].mean()) if ng.any() else np.nan,
                         S_other=float(v["t"][~ng].mean()) if (~ng).any() else np.nan,
                         fail=fail.get(key, np.nan)))
    print(f"determinism: max |clean_repeat - clean| = {det:.3g}; clean action std trans {sd_t:.4f} rot {sd_r:.4f}")
    print("| policy | task | queries | S | S90 | rot | grip flip | S near grasp | S other | closed-loop fail |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['policy']} | {r['suite'].replace('libero_', '')} {r['task']} | {r['n_queries']} | {r['S']:.3f} | "
              f"{r['S90']:.3f} | {r['R']:.3f} | {r['G']:.3f} | {r['S_grasp']:.3f} | {r['S_other']:.3f} | {100 * r['fail']:.0f}% |")
    f = np.array([r["fail"] for r in rows])
    res = {}
    for m in ("S", "S90", "R", "G", "S_grasp"):
        x = np.array([r[m] for r in rows])
        ok = ~np.isnan(x)
        rho, p = spearmanr(x[ok], f[ok])
        res[m] = (float(rho), float(p))
        print(f"Spearman({m}, fail) = {rho:+.2f}, p = {p:.3g}")
    by = {(r["policy"], r["suite"], r["task"]): r for r in rows}
    hits = []
    for t, hi, lo in CROSS:
        a, b = by[(hi, *t)]["S"], by[(lo, *t)]["S"]
        hits.append(a > b)
        print(f"cross {t[0]}:{t[1]}: S({hi}) = {a:.3f} vs S({lo}) = {b:.3f} -> {'ok' if a > b else 'WRONG'}")
    rho, p = res["S"]
    nh = sum(hits)
    if rho >= 0.5 and p < 0.05 and nh >= 3:
        v = "H1 SUPPORTED"
    elif rho < 0.3 or nh <= 2:
        v = "H1 NOT SUPPORTED"
    else:
        v = "PARTIAL"
    print(f"Rule: {v} (rho {rho:+.2f}, p {p:.3g}, cross {nh}/4)")
    json.dump({"rows": rows, "spearman": res, "cross_hits": nh, "verdict": v, "determinism_max": det},
              open(os.path.join(os.path.dirname(PROBE), "summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
