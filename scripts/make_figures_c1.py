"""Figures for REPORT_C1.md, drawn from the shipped per-episode CSVs, results/e010/summary.json and the
first-query frames in results/e009/c1_frames.npz (LIBERO Goal 0, init state 20).

  python scripts/make_figures_c1.py         # writes figures/fig3_c1.{png,pdf}, figures/fig4_c1_fragility.{png,pdf}

Needs numpy, scipy and matplotlib. [ai-scaffold]
"""
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_figures import C_BAD, C_GREY, C_OFT, C_OFTW, wilson  # noqa: E402  (also sets the rcParams)

RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")
EXPOSURES = [4, 8, 16, 32, 48]
FRAGILE = [("libero_goal", "0"), ("libero_object", "0")]


def counts(path, by=("condition",), where=None):
    kn = defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(path)):
        if int(r["error"]) or (where and not where(r)):
            continue
        k = tuple(r[b] for b in by)
        kn[k][0] += int(r["success"])
        kn[k][1] += 1
    return kn


def errbar(ax, x, kn, key, color, marker="o", label=None, ls="-", ms=6):
    pts = [(xi, *wilson(*kn[k])[:3]) for xi, k in zip(x, key) if kn[k][1]]
    xs, p, lo, hi = map(np.array, zip(*pts))
    ax.errorbar(xs, p, yerr=[p - lo, hi - p], color=color, marker=marker, ls=ls, ms=ms, capsize=3, label=label)


def fig3():
    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.32)
    top = gs[0].subgridspec(2, 6, wspace=0.04, hspace=0.06)
    fr = np.load(os.path.join(RES, "e009", "c1_frames.npz"))
    cols = [("base", "clean"), ("t32", "32 ms"), ("t16", "16 ms"), ("t8", "8 ms"), ("t4", "4 ms"),
            ("u8_noise4", "8-bit path,\n4 ms noise")]
    for j, (arm, title) in enumerate(cols):
        for i, cam in enumerate(("agentview", "wrist")):
            ax = fig.add_subplot(top[i, j])
            ax.imshow(fr[f"{arm}/{cam}"][::-1, ::-1])          # LIBERO renders are stored upside down
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=9, fontweight="normal", loc="center")
            if j == 0:
                ax.set_ylabel(cam, fontsize=9)
    fig.text(0.1, 0.935, "(a) First policy frame, Goal 0, at each exposure (RAW path unless noted)",
             fontweight="bold", fontsize=11)

    bot = gs[1].subgridspec(1, 2, wspace=0.28, width_ratios=[1.25, 1])
    ax = fig.add_subplot(bot[0])
    oft = counts(os.path.join(RES, "e009", "oft.csv"))
    oftw = counts(os.path.join(RES, "e009", "oft_w.csv"))
    errbar(ax, EXPOSURES, oft, [(f"t{t}",) for t in EXPOSURES], C_OFT, label="OFT: blur + noise")
    errbar(ax, [4, 16, 32], oftw, [(f"t{t}",) for t in (4, 16, 32)], C_OFTW, label="OFT_w: blur + noise")
    errbar(ax, [4.4], oft, [("noise4",)], C_BAD, marker="s", ls="none", label="OFT: noise only (4 ms gain)")
    errbar(ax, [32.8], oft, [("blur32",)], C_GREY, marker="D", ls="none", label="OFT: blur only (32 ms)")
    b = wilson(*oft[("base",)])[0]
    ax.axhline(b, color=C_OFT, lw=0.8, ls=":")
    ax.text(48.5, b + 0.6, "OFT clean", color=C_OFT, fontsize=8, ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xticks(EXPOSURES, [str(t) for t in EXPOSURES])
    ax.set_xlabel("exposure time (ms), gain ∝ 1/t")
    ax.set_ylabel("success (%)")
    ax.set_ylim(70, 101)
    ax.set_title("(b) Success vs exposure, 9 tasks × 20")
    ax.legend(fontsize=8, loc="lower right", frameon=False)

    ax = fig.add_subplot(bot[1])
    tasks = [f"{s}:{t}" for s in ("libero_spatial", "libero_object", "libero_goal") for t in "012"]
    by_t = counts(os.path.join(RES, "e009", "oft.csv"), by=("condition", "suite", "task_id"))
    by_tw = counts(os.path.join(RES, "e009", "oft_w.csv"), by=("condition", "suite", "task_id"))
    rows = [("OFT clean", by_t, "base"), ("OFT 4 ms", by_t, "t4"), ("OFT_w clean", by_tw, "base"),
            ("OFT_w 4 ms", by_tw, "t4")]
    M = np.array([[100 * d[(c, *t.split(":"))][0] / d[(c, *t.split(":"))][1] for t in tasks] for _, d, c in rows])
    im = ax.imshow(M, cmap="RdYlBu", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] < 45 else "black")
    ax.set_xticks(range(len(tasks)), [t.replace("libero_", "").replace(":", " ") for t in tasks], rotation=60,
                  fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.set_title("(c) Success by task (%), n = 20 per cell")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig3_c1.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig4():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.3, "width_ratios": [0.8, 1.2]})
    ax = axs[0]
    for seed, d in enumerate(("e009", "e009_s1", "e009_s2")):
        kn = counts(os.path.join(RES, d, "oft.csv"),
                    where=lambda r: (r["suite"], r["task_id"]) in FRAGILE and 20 <= int(r["episode"]) < 40)
        for x0, arm, col in ((0, "noise4", C_BAD), (1, "u8_noise4", C_GREY)):
            k, n = kn[(arm,)]
            p, lo, hi = wilson(k, n)
            ax.errorbar(x0 + 0.12 * (seed - 1), p, yerr=[[p - lo], [hi - p]], color=col, marker="os^"[seed],
                        capsize=3, ls="none", label=f"seed {seed}" if x0 == 0 else None)
    ax.set_xticks([0, 1], ["RAW path\n(noise4)", "8-bit path\n(u8_noise4)"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) RAW vs 8-bit path, Goal 0 + Object 0\n4 ms noise, per noise seed (n = 40 each)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax = axs[1]
    s = json.load(open(os.path.join(RES, "e010", "summary.json")))
    for r in s["rows"]:
        col = C_OFT if r["policy"] == "oft" else C_OFTW
        ax.scatter(r["S"], 100 * r["fail"], color=col, s=30)
        if r["fail"] >= 0.25:
            ax.annotate(f'{"OFT" if r["policy"] == "oft" else "OFT_w"} {r["suite"].replace("libero_", "")} {r["task"]}', (r["S"], 100 * r["fail"]), fontsize=8,
                        xytext=(4, 2), textcoords="offset points")
    ax.scatter([], [], color=C_OFT, label="OFT")
    ax.scatter([], [], color=C_OFTW, label="OFT_w")
    rho, p = s["spearman"]["S"]
    ax.set_xlabel("open-loop sensitivity S (normalised action change under 4 ms noise)")
    ax.set_ylabel("closed-loop failure at 4 ms (%)")
    ax.set_title(f"(b) Open-loop sensitivity vs closed-loop failure\n18 policy × task cells (Spearman ρ = {rho:+.2f}, p = {p:.2f})")
    ax.legend(fontsize=8, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig4_c1_fragility.{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig3()
    fig4()
