"""E007 summary: blur length distributions over policy frames, and the stop rule.

Reads results/e007/frames.csv (e007/motion.py). Per camera, the per-frame blur value is
  wrist: median |dpx| over valid pixels (rigid-body flow; static-scene flow reported alongside),
  agentview: EEF-site |dpx| (dense rigid-flow per-frame p99 reported as secondary),
scaled to exposure t_e by blur = |dpx per step| * t_e * control_freq (control_freq = 20 Hz, logged).
Stop rule (notebook E007): FAIL if p95 over frames of the blur at 33 ms < 1 px for BOTH cameras.
[ai-scaffold]
"""
import csv
import json
import os
import sys

import numpy as np

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results", "e007")
HZ = 20.0
EXPOSURES_MS = (8, 16, 33)
METRICS = {"wrist (median px, rigid)": "wrist_med", "wrist (median px, static)": "wrist_static_med",
           "wrist (p95 px, rigid)": "wrist_p95", "agentview EEF": "agent_eef",
           "agentview (p99 px, rigid)": "agent_p99"}


def stats(x):
    x = np.asarray(x, float)
    return dict(n=len(x), median=float(np.median(x)), p95=float(np.percentile(x, 95)), share_ge1=float((x >= 1).mean()))


def main():
    rows = list(csv.DictReader(open(os.path.join(DIR, "frames.csv"))))
    pol = [r for r in rows if r["kind"] == "policy"]
    settle = [r for r in rows if r["kind"] == "settle"]
    out = {"n_episodes": len({(r["suite"], r["task"], r["ep"]) for r in rows}), "n_policy_frames": len(pol),
           "exposure": {}, "by_phase_33ms": {}, "by_suite_33ms": {}, "settle_33ms": {}}
    for te in EXPOSURES_MS:
        k = te * 1e-3 * HZ
        out["exposure"][f"{te}ms"] = {name: stats([float(r[c]) * k for r in pol]) for name, c in METRICS.items()}
    k33 = 33e-3 * HZ
    for ph in ("approach", "grasp", "transport"):
        sub = [r for r in pol if r["phase"] == ph]
        out["by_phase_33ms"][ph] = {name: stats([float(r[c]) * k33 for r in sub]) for name, c in METRICS.items()}
    for s in sorted({r["suite"] for r in pol}):
        sub = [r for r in pol if r["suite"] == s]
        out["by_suite_33ms"][s] = {name: stats([float(r[c]) * k33 for r in sub]) for name, c in METRICS.items()}
    out["settle_33ms"] = {name: stats([float(r[c]) * k33 for r in settle]) for name, c in METRICS.items()}
    e33 = out["exposure"]["33ms"]
    wrist_p95, agent_p95 = e33["wrist (median px, rigid)"]["p95"], e33["agentview EEF"]["p95"]
    out["rule"] = dict(wrist_p95_33ms=wrist_p95, agentview_eef_p95_33ms=agent_p95,
                       wrist_pass=wrist_p95 >= 1, agentview_pass=agent_p95 >= 1,
                       verdict="PASS" if (wrist_p95 >= 1 or agent_p95 >= 1) else "FAIL")
    json.dump(out, open(os.path.join(DIR, "summary.json"), "w"), indent=1)

    print(f"episodes {out['n_episodes']}, policy frames {out['n_policy_frames']}")
    for te, block in out["exposure"].items():
        print(f"\n== t_e = {te}: median / p95 / share >= 1 px")
        for name, s in block.items():
            print(f"  {name:28s} {s['median']:6.2f} {s['p95']:6.2f} {s['share_ge1']:6.1%}")
    for title, key in (("phase", "by_phase_33ms"), ("suite", "by_suite_33ms")):
        print(f"\n== 33 ms by {title}: n, wrist median-px p95, agentview EEF p95, EEF median")
        for g, block in out[key].items():
            print(f"  {g:16s} {block['agentview EEF']['n']:5d} {block['wrist (median px, rigid)']['p95']:6.2f} "
                  f"{block['agentview EEF']['p95']:6.2f} {block['agentview EEF']['median']:6.2f}")
    print("\nrule:", out["rule"])


if __name__ == "__main__":
    main()
