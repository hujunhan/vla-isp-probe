"""Aggregate E002 per-episode CSVs and compare with LIBERO-Plus Table 10 / Table 2 (OpenVLA-OFT row).

Metric definitions (decisions/004):
  * success rate = successes / valid episodes, where valid = episodes without an "Episode error".
    Errors are reported separately and never counted as policy failures.
  * pooling: the paper's "Avg" rows equal the task-count-weighted (pooled) rate over all tasks of a
    category, not the mean of the four suite rates (checked: pooled Noise 75.78, Light 88.69), so we
    pool the same way.
  * 95% CI of our rate: Wilson interval.
  * match test vs the paper: difference d = ours - paper; the paper's number is itself a 1-episode-per-
    task binomial estimate over the same n tasks, so SE(d) = sqrt(p1(1-p1)/n1 + p2(1-p2)/n2) with
    n2 = number of tasks in that cell of the benchmark. Verdict:
      match       |d| <= 1.96 * SE(d)
      diverge     |d| > 10 points (stopping rule S2 threshold)
      ambiguous   otherwise (int4 vs bf16 run decides)

Usage: python aggregate_e002.py results/*.csv [--markdown]

Advisor-written metric code (decisions/004): owner explain-back owed before public use.
"""

import csv
import math
import sys
from collections import defaultdict

# LIBERO-Plus arXiv 2510.13626, Table 10 (p31), OpenVLA-OFT rows; Table 2 (p9) for the pooled row.
PAPER = {
    ("libero_spatial", "Light Conditions"): 98.3, ("libero_spatial", "Sensor Noise"): 96.3,
    ("libero_object", "Light Conditions"): 73.7, ("libero_object", "Sensor Noise"): 72.3,
    ("libero_goal", "Light Conditions"): 93.9, ("libero_goal", "Sensor Noise"): 75.2,
    ("libero_10", "Light Conditions"): 89.4, ("libero_10", "Sensor Noise"): 63.5,
    ("ALL", "Light Conditions"): 88.7, ("ALL", "Sensor Noise"): 75.8,
}
# Number of tasks per cell in task_classification.json (= paper's n, one episode per task).
BENCH_N = {
    ("libero_spatial", "Light Conditions"): 292, ("libero_spatial", "Sensor Noise"): 351,
    ("libero_object", "Light Conditions"): 297, ("libero_object", "Sensor Noise"): 422,
    ("libero_goal", "Light Conditions"): 279, ("libero_goal", "Sensor Noise"): 379,
    ("libero_10", "Light Conditions"): 274, ("libero_10", "Sensor Noise"): 449,
    ("ALL", "Light Conditions"): 1142, ("ALL", "Sensor Noise"): 1601,
}
NOISE_FAMILY = {0: "motion blur", 1: "gaussian blur", 2: "zoom blur", 3: "fog", 4: "glass blur"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * (centre - half), 100 * (centre + half)


class Cell:
    def __init__(self):
        self.n = self.k = self.err = 0

    def add(self, row):
        if int(row["error"]):
            self.err += 1
        else:
            self.n += 1
            self.k += int(row["success"])

    @property
    def rate(self):
        return 100 * self.k / self.n if self.n else float("nan")


def verdict(ours, n1, paper, n2):
    p1, p2 = ours / 100, paper / 100
    se = 100 * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if n1 else float("nan")
    d = ours - paper
    if abs(d) <= 1.96 * se:
        v = "match"
    elif abs(d) > 10:
        v = "diverge"
    else:
        v = "ambiguous"
    return d, se, v


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p from discordant counts b (clean-only) and c (raw-only)."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(csv.DictReader(f))
    return rows


def main(paths):
    rows = load(paths)
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang_mode"]].append(r)

    for lang, rs in sorted(by_lang.items()):
        print(f"\n## lang_mode = {lang}  ({len(rs)} episodes)\n")

        # 1. Suite x category, plus pooled, vs paper
        cells = defaultdict(Cell)
        for r in rs:
            cells[(r["suite"], r["category"])].add(r)
            cells[("ALL", r["category"])].add(r)
        print("| Suite | Category | done / bench n | errors | ours % [95% CI] | paper % | diff | verdict |")
        print("|---|---|---|---|---|---|---|---|")
        for key in sorted(cells, key=lambda k: (k[0] == "ALL", k)):
            c = cells[key]
            lo, hi = wilson(c.k, c.n)
            bench_n = BENCH_N.get(key, 0)
            complete = (c.n + c.err) == bench_n
            if key in PAPER:
                d, se, v = verdict(c.rate, c.n, PAPER[key], bench_n)
                if not complete:
                    v = "— (partial)"
                elif c.k in (0, c.n):
                    v += " (ceiling: our variance term is 0)"
                paper_s, diff_s = f"{PAPER[key]:.1f}", f"{d:+.1f} (±{1.96 * se:.1f})"
            else:
                paper_s = diff_s = v = "—"
            print(f"| {key[0]} | {key[1]} | {c.n + c.err} / {bench_n} | {c.err} | "
                  f"{c.rate:.1f} [{lo:.1f}, {hi:.1f}] | {paper_s} | {diff_s} | {v} |")

        # 2. Outcome-defined levels (#6 App. C.3, p18): L_k = solved by (5-k) of 4 models incl. OFT.
        #    NOT a severity axis. L1 => paper's OFT succeeded, L5 => paper's OFT failed, so the L1 and
        #    L5 columns are a per-task agreement check with the paper's single episodes.
        print("\nOutcome-defined levels (not severity). Paper OFT: 100% on L1, 0% on L5.")
        print("\n| Category | Suite | L1 | L2 | L3 | L4 | L5 |")
        print("|---|---|---|---|---|---|---|")
        lv = defaultdict(Cell)
        for r in rs:
            for s in (r["suite"], "ALL"):
                lv[(r["category"], s, r["level"])].add(r)
        for cat in sorted({r["category"] for r in rs}):
            for s in sorted({r["suite"] for r in rs}) + ["ALL"]:
                cols = []
                for L in ["1", "2", "3", "4", "5"]:
                    c = lv.get((cat, s, L))
                    cols.append(f"{c.rate:.0f} (n={c.n})" if c and c.n else "—")
                print(f"| {cat} | {s} | " + " | ".join(cols) + " |")

        # 3. Noise family
        fam = defaultdict(Cell)
        for r in rs:
            if r["category"] == "Sensor Noise" and r["noise_code"]:
                fam[NOISE_FAMILY[(int(r["noise_code"]) - 1) // 10]].add(r)
        if fam:
            print("\n| Noise family | n | success % [95% CI] |")
            print("|---|---|---|")
            for name, c in sorted(fam.items(), key=lambda kv: kv[1].rate):
                lo, hi = wilson(c.k, c.n)
                print(f"| {name} | {c.n} | {c.rate:.1f} [{lo:.1f}, {hi:.1f}] |")

            # 3b. Severity within family (code 1..10 inside each block of 10): the physical trend axis.
            sev = defaultdict(Cell)
            for r in rs:
                if r["category"] == "Sensor Noise" and r["noise_code"]:
                    code = int(r["noise_code"])
                    sev[(NOISE_FAMILY[(code - 1) // 10], (code - 1) % 10 + 1)].add(r)
            print("\n| Noise family | " + " | ".join(f"s{i}" for i in range(1, 11)) + " |")
            print("|---|" + "---|" * 10)
            for name in NOISE_FAMILY.values():
                cols = []
                for i in range(1, 11):
                    c = sev.get((name, i))
                    cols.append(f"{c.rate:.0f} ({c.n})" if c and c.n else "—")
                print(f"| {name} | " + " | ".join(cols) + " |")

    # 4. Paired instruction ablation: same (suite, task_id) under clean and raw.
    clean = {(r["suite"], r["task_id"]): r for r in by_lang.get("clean", [])}
    raw = {(r["suite"], r["task_id"]): r for r in by_lang.get("raw", [])}
    keys = sorted(set(clean) & set(raw))
    if keys:
        print("\n## Paired instruction ablation (clean vs raw, same tasks; errors excluded)\n")
        print("| Suite | Category | pairs | clean succ | raw succ | clean-only | raw-only | McNemar p |")
        print("|---|---|---|---|---|---|---|---|")
        groups = defaultdict(list)
        for k in keys:
            a, b = clean[k], raw[k]
            if int(a["error"]) or int(b["error"]):
                continue
            groups[(k[0], a["category"])].append((int(a["success"]), int(b["success"])))
        for (suite, cat), pr in sorted(groups.items()):
            b, c = sum(a > r for a, r in pr), sum(r > a for a, r in pr)
            print(f"| {suite} | {cat} | {len(pr)} | {sum(a for a, _ in pr)} | {sum(r for _, r in pr)} | "
                  f"{b} | {c} | {mcnemar_exact(b, c):.3g} |")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")])
