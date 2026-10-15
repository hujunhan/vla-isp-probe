"""E008 aggregation: success per condition with Wilson 95% CIs, per suite, and the pre-registered rule.

Rule (notebook experiments/E008-filament-renderer-gap.md):
  pass  if the best Filament condition is within 10 points of `classic`; C1 may use the HDR path only if
        `fil_raw_isp` itself is within 10 points
  fail  if every Filament condition drops by > 10 points
  5-10 point drops are reported as a confound for the owner to judge.
python e008/aggregate_e008.py [results/e008/oft.csv]
[ai-scaffold]
"""
import csv
import math
import sys
from collections import defaultdict

ORDER = ["classic", "fil_post", "fil_hdr_oetf", "fil_raw_isp"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/e008/oft.csv"
    rows = list(csv.DictReader(open(path)))
    tot, suite, errs = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), defaultdict(int)
    for r in rows:
        s = int(r["success"])
        tot[r["condition"]][0] += s
        tot[r["condition"]][1] += 1
        suite[(r["condition"], r["suite"])][0] += s
        suite[(r["condition"], r["suite"])][1] += 1
        errs[r["condition"]] += int(r["error"])
    suites = sorted({r["suite"] for r in rows})
    ref = 100 * tot["classic"][0] / max(tot["classic"][1], 1)
    print("| Condition | n | Success [95% CI] | Drop vs classic | " + " | ".join(s.replace("libero_", "") for s in suites) + " | Errors |")
    print("|---|---|---|---|" + "---|" * len(suites) + "---|")
    drops = {}
    for c in ORDER:
        k, n = tot[c]
        if n == 0:
            continue
        p = 100 * k / n
        lo, hi = wilson(k, n)
        drops[c] = ref - p
        per = " | ".join(f"{suite[(c, s)][0]}/{suite[(c, s)][1]}" for s in suites)
        print(f"| {c} | {n} | {p:.1f} [{lo:.1f}, {hi:.1f}] | {'—' if c == 'classic' else f'{ref - p:+.1f}'} | {per} | {errs[c]} |")
    fil = {c: d for c, d in drops.items() if c != "classic"}
    if fil:
        best = min(fil, key=fil.get)
        verdict = "PASS" if fil[best] <= 10 else "FAIL"
        print(f"\nBest Filament condition: {best} (drop {fil[best]:+.1f}). Rule: {verdict}.")
        if "fil_raw_isp" in fil:
            print(f"fil_raw_isp drop {fil['fil_raw_isp']:+.1f} -> HDR path usable for C1: {'yes' if fil['fil_raw_isp'] <= 10 else 'no'}")
        for c, d in fil.items():
            if 5 <= d <= 10:
                print(f"note: {c} drop {d:+.1f} is in the 5-10 band (confound; owner judges)")


if __name__ == "__main__":
    main()
