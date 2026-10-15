"""E009 Stage A aggregation: success per exposure with Wilson 95% CIs, drop vs the clean baseline, one-sided
Fisher p (condition worse than base), per suite, and the pre-registered rule (notebook E009):
  effect : some exposure drops >= 10 points vs base with Fisher p < 0.05 -> Stage B
  null   : all exposures within 5 points of base
  5-10   : owner's call
python e009/aggregate_e009.py [results/e009/oft.csv]
[ai-scaffold]
"""
import csv
import sys
from collections import defaultdict

from scipy.stats import fisher_exact

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/e008")
from aggregate_e008 import wilson  # noqa: E402

ORDER = ["base", "t4", "t8", "t16", "t32", "t48", "blur32", "noise4", "img32", "noise4_signed",
         "u8_clean", "u8_e005", "u8_noise4"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/e009/oft.csv"
    rows = list(csv.DictReader(open(path)))
    tot, suite, errs = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), defaultdict(int)
    for r in rows:
        s = int(r["success"])
        for key, d in ((r["condition"], tot), ((r["condition"], r["suite"]), suite)):
            d[key][0] += s
            d[key][1] += 1
        errs[r["condition"]] += int(r["error"])
    suites = sorted({r["suite"] for r in rows})
    kb, nb = tot["base"]
    ref = 100 * kb / max(nb, 1)
    print("| Condition | n | Success [95% CI] | Drop vs base | Fisher p (one-sided) | "
          + " | ".join(s.replace("libero_", "") for s in suites) + " | Errors |")
    print("|---|---|---|---|---|" + "---|" * len(suites) + "---|")
    drops = {}
    for c in ORDER:
        k, n = tot[c]
        if n == 0:
            continue
        p = 100 * k / n
        lo, hi = wilson(k, n)
        pv = "—" if c == "base" else f"{fisher_exact([[kb, nb - kb], [k, n - k]], alternative='greater')[1]:.3g}"
        if c != "base":
            drops[c] = (ref - p, float(pv))
        per = " | ".join(f"{suite[(c, s)][0]}/{suite[(c, s)][1]}" for s in suites)
        print(f"| {c} | {n} | {p:.1f} [{lo:.1f}, {hi:.1f}] | {'—' if c == 'base' else f'{ref - p:+.1f}'} | {pv} | {per} | {errs[c]} |")
    if drops:
        if any(d >= 10 and pv < 0.05 for d, pv in drops.values()):
            verdict = "EFFECT (>= 10 points, p < 0.05) -> Stage B"
        elif all(abs(d) < 5 for d, _ in drops.values()):
            verdict = "NULL (all within 5 points)"
        else:
            verdict = "5-10 BAND (owner's call)"
        print(f"\nRule (Stage A): {verdict}")
        stage_b(tot)
        stage_c(tot)


def stage_b(tot):
    """Stage B rules (notebook E009, frozen 2026-10-12)."""
    def fisher(a, b, alt):
        (ka, na), (kb, nb) = tot[a], tot[b]
        return fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative=alt)[1]

    def pct(c):
        return 100 * tot[c][0] / tot[c][1]
    if tot["blur32"][1] or tot["t48"][1]:
        hits = [c for c in ("blur32", "t48") if tot[c][1] and pct("base") - pct(c) >= 5 and fisher("base", c, "greater") < 0.05]
        print("Blur matters:", f"yes ({', '.join(hits)})" if hits else "no (blur <= ~5 points at LIBERO speeds)")
    if tot["noise4"][1] and tot["t4"][1]:
        ok = abs(pct("noise4") - pct("t4")) < 5 and pct("base") - pct("noise4") >= 10
        print(f"4 ms drop is noise: {'yes' if ok else 'no'} (noise4 {pct('noise4'):.1f} vs t4 {pct('t4'):.1f})")
    if tot["img32"][1] and tot["blur32"][1]:
        d, pv = pct("img32") - pct("blur32"), fisher("img32", "blur32", "two-sided")
        print(f"Physical vs image-space blur: img32 - blur32 = {d:+.1f}, p = {pv:.3g} -> {'differ' if abs(d) >= 10 and pv < 0.05 else 'no difference'}")


def stage_c(tot):
    """Stage C rules (notebook E009, frozen 2026-10-12, decisions/014)."""
    def fisher(a, b, alt):
        (ka, na), (kb, nb) = tot[a], tot[b]
        return fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative=alt)[1]

    def pct(c):
        return 100 * tot[c][0] / tot[c][1]
    have = lambda *cs: all(tot[c][1] for c in cs)  # noqa: E731
    if have("u8_noise4", "noise4"):
        d, pv = pct("u8_noise4") - pct("noise4"), fisher("u8_noise4", "noise4", "two-sided")
        v = "RAW path changes the result" if abs(d) >= 10 and pv < 0.05 else (
            "8-bit path suffices at matched noise" if abs(d) < 5 else "inconclusive")
        print(f"Q1 path (levels): u8_noise4 - noise4 = {d:+.1f}, p = {pv:.3g} -> {v}")
        if have("u8_clean", "base"):
            dc = pct("u8_clean") - pct("base")
            dd = (pct("u8_clean") - pct("u8_noise4")) - (pct("base") - pct("noise4"))
            print(f"   clean refs: u8_clean - base = {dc:+.1f}" + (f" (>= 5: compare drops; u8 drop - RAW drop = {dd:+.1f})" if abs(dc) >= 5 else ""))
    if have("u8_e005", "u8_noise4"):
        d, pv = pct("u8_e005") - pct("u8_noise4"), fisher("u8_e005", "u8_noise4", "greater")
        print(f"Q2 noise model: u8_e005 - u8_noise4 = {d:+.1f}, p = {pv:.3g} -> "
              f"{'noise model explains E005 vs E009' if d >= 10 and pv < 0.05 else 'not explained by the noise model alone'}")
    if have("noise4_signed", "noise4"):
        d, pv = pct("noise4_signed") - pct("noise4"), fisher("noise4_signed", "noise4", "greater")
        v = "major" if d >= 7 and pv < 0.05 else ("minor" if d < 5 else "partial")
        print(f"Q3 rectification before NR: noise4_signed - noise4 = {d:+.1f}, p = {pv:.3g} -> {v}")


if __name__ == "__main__":
    main()
