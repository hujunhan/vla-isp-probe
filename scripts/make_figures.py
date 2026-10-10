"""Figures for REPORT.md, drawn from the shipped per-episode CSVs (and, for the image panel, the camera
model applied to a LIBERO render in results/e004/frames.npz).

  python scripts/make_figures.py            # writes figures/fig1_isp.{png,pdf}, figures/fig2_suffix.{png,pdf}

Needs numpy + matplotlib; the image panel also needs Infinite-ISP (INFINITE_ISP_ROOT). [ai-scaffold]
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")

# Okabe-Ito colour-blind-safe palette
C_OFT, C_OFTW, C_GREY, C_BAD = "#0072B2", "#E69F00", "#999999", "#D55E00"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
                     "savefig.dpi": 200, "font.family": "DejaVu Sans"})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def rates(path, key="condition", where=None):
    """{key value: (rate, lo, hi, n)} over valid (non-error) episodes."""
    kn = defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(path)):
        if int(r["error"]) or (where and not where(r)):
            continue
        kn[r[key]][0] += int(r["success"])
        kn[r[key]][1] += 1
    return {c: (*wilson(k, n), n) for c, (k, n) in kn.items()}


def image_panel(fig, gs_row):
    """Row of camera images: render + Infinite-ISP presets at lambda = 0.012 (E005 configs), cropped."""
    from isp.camera import read_noise_for
    from isp.infinite import preset, simulate_infinite

    frames = np.load(os.path.join(RES, "e004", "frames.npz"))
    i = list(frames["meta"]).index("libero_object:1:0")
    render = frames["agentview"][i]
    conds = {c["tag"]: c for b in json.load(open(os.path.join(RES, "e005", "conditions.json")))["blocks"].values()
             for c in b}

    def cam(tag):
        c = conds[tag]
        p = preset(c["preset"], bnr_scale=c.get("bnr_scale", 1.0), wts=c.get("wts", 10.0))
        return simulate_infinite(render, c["lam"], read_noise_for(c["lam"]), p, np.random.default_rng(0))

    view = lambda im: im[::-1, ::-1]           # LIBERO agentview is stored upside down
    y0, y1, x0, x1 = 70, 160, 60, 230          # crop (in the displayed orientation)
    tiles = [("Rendered\n(training images)", view(render), "k"),
             ("NR off", view(cam("inf_lam1.2e-2_off")), C_BAD),
             ("Repo-default NR", view(cam("inf_lam1.2e-2_repo")), C_BAD),
             ("Tuned NR", view(cam("inf_lam1.2e-2_tuned")), "#009E73"),
             ("Strong NR (2× tuned)", view(cam("inf_lam1.2e-2_strong")), "k")]
    sub = gs_row.subgridspec(1, len(tiles) + 1, wspace=0.06, width_ratios=[1.25] + [1] * len(tiles))
    ax = fig.add_subplot(sub[0])
    ax.imshow(tiles[0][1])
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1.5, ls="--"))
    ax.set_title("(a) What the policy sees: high sensor gain (λ = 0.012) through Infinite-ISP",
                 loc="left", x=0, pad=8)
    ax.set_xlabel("Full third-person view", fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])
    for j, (name, im, col) in enumerate(tiles):
        a = fig.add_subplot(sub[j + 1])
        a.imshow(im[y0:y1, x0:x1], interpolation="nearest")
        a.set_xticks([]), a.set_yticks([])
        a.set_xlabel(name, fontsize=9, color=col, fontweight="bold" if col != "k" else "normal")
        for s in a.spines.values():
            s.set_visible(True), s.set_color(col), s.set_linewidth(2 if col != "k" else 0.6)


def isp_bars(ax):
    """E005 success by Infinite-ISP preset, both policies; E006 fresh-episode points for OFT."""
    e5 = {p: rates(os.path.join(RES, "e005", f"{p}.csv")) for p in ("oft", "oft_w")}
    e6 = rates(os.path.join(RES, "e006", "oft.csv"))
    groups = [("Clean\ncamera", "inf_clean_sharp"), ("NR off", "inf_lam1.2e-2_off"),
              ("Repo-default\nNR", "inf_lam1.2e-2_repo"), ("Tuned\nNR", "inf_lam1.2e-2_tuned"),
              ("Strong\nNR", "inf_lam1.2e-2_strong")]
    x = np.arange(len(groups))
    w = 0.38
    ax.axvspan(0.5, 2.5, color=C_BAD, alpha=0.07, lw=0)
    ax.text(1.5, 104, "un-tuned NR", ha="center", color=C_BAD, fontsize=9, fontweight="bold")
    for k, (pol, col, lab) in enumerate([("oft", C_OFT, "OFT (wrist + third-person cameras)"),
                                         ("oft_w", C_OFTW, "OFT_w (third-person camera only)")]):
        r = [e5[pol][t] for _, t in groups]
        xs = x + (k - 0.5) * w
        ax.bar(xs, [v[0] for v in r], w, color=col, label=lab,
               yerr=[[v[0] - v[1] for v in r], [v[2] - v[0] for v in r]], capsize=2.5,
               error_kw={"lw": 0.9, "ecolor": "#333"})
        for xi, v in zip(xs, r):
            ax.text(xi, v[1] - 1.5, f"{v[0]:.0f}", ha="center", va="top", fontsize=8, color="white",
                    fontweight="bold")
    # E006: OFT on fresh initial states (NR off, repo default)
    for gi, tag in ((1, "inf_lam1.2e-2_off"), (2, "inf_lam1.2e-2_repo")):
        v = e6[tag]
        ax.errorbar(gi - 0.5 * w, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="D", ms=5, mfc="white",
                    mec=C_OFT, mew=1.5, ecolor=C_OFT, capsize=2, lw=0.9,
                    label="OFT, replication on fresh episodes" if gi == 1 else None, zorder=5)
    ax.set_xticks(x, [g for g, _ in groups])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Task success (%)")
    ax.set_title("(b) Closed-loop success vs. ISP noise reduction (NR)", pad=8)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.95)
    ax.text(0.99, -0.24, "9 LIBERO tasks × 20 episodes (diamonds: × 30, new init states); 95% Wilson CIs.\nNR off is compared with a clean camera without sharpening (OFT 99, OFT_w 97).",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="#555")


def raw_vs_imgspace(ax):
    """E004: RAW noise through the toy ISP (tuned NR) vs image-space noise at matched PSNR."""
    levels = [(1e-3, "lam1e-3", "img_sigma_lam1e-3"), (5e-3, "lam5e-3", "img_sigma_lam5e-3"),
              (1.2e-2, "lam1.2e-2", "img_sigma_lam1.2e-2"), (3e-2, "stress_lam3e-2", "img_sigma_lam3e-2")]
    lam = [l for l, _, _ in levels]
    ax.axvspan(1.7e-2, 4.5e-2, color=C_GREY, alpha=0.15, lw=0)
    ax.text(2.75e-2, 9, "beyond\nplausible\n(stress)", ha="center", fontsize=8, color="#555")
    for pol, col, name in (("oft", C_OFT, "OFT"), ("oft_w", C_OFTW, "OFT_w")):
        r = rates(os.path.join(RES, "e004", f"{pol}.csv"))
        for idx, ls, mk, kind in ((1, "-", "o", "RAW noise → ISP"), (2, "--", "s", "image-space noise")):
            v = [r[t[idx]] for t in levels]
            ax.errorbar(lam, [a[0] for a in v], yerr=[[a[0] - a[1] for a in v], [a[2] - a[0] for a in v]],
                        color=col, ls=ls, marker=mk, ms=5, mfc=col if ls == "-" else "white", capsize=2,
                        lw=1.6, label=f"{name}: {kind}")
    ax.set_xscale("log")
    ax.set_xlim(7e-4, 4.5e-2)
    ax.set_ylim(0, 106)
    ax.set_xlabel("Sensor noise level λ (shot-noise gain, log scale)")
    ax.set_ylabel("Task success (%)")
    ax.set_title("(c) Same PSNR, different outcome: RAW vs. image-space noise", pad=8)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.95)
    ax.text(0.99, -0.2, "Toy ISP with tuned NR; 3 tasks × 50 episodes; 95% Wilson CIs",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="#555")


def fig1():
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.55], hspace=0.32, top=0.9)
    image_panel(fig, gs[0])
    bot = gs[1].subgridspec(1, 2, wspace=0.22)
    isp_bars(fig.add_subplot(bot[0]))
    raw_vs_imgspace(fig.add_subplot(bot[1]))
    fig.suptitle("Realistic sensor noise is harmless when the ISP's noise reduction is tuned; un-tuned noise reduction is not",
                 fontsize=13, fontweight="bold", x=0.06, ha="left", y=0.97)
    save(fig, "fig1_isp")


def fig2():
    noise = lambda r: r["category"] == "Sensor Noise"
    clean = rates(os.path.join(RES, "e002", "goal_clean.csv"), key="category", where=noise)["Sensor Noise"]
    raw_rows = [r for p in ("goal_raw_full_a.csv", "goal_raw_full_b.csv")
                for r in csv.DictReader(open(os.path.join(RES, "e003", p))) if not int(r["error"])]
    k = sum(int(r["success"]) for r in raw_rows)
    raw = (*wilson(k, len(raw_rows)), len(raw_rows))
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    rows = [("Clean instruction (ours)", clean, C_OFT),
            ("Suffixed instruction (ours)", raw, C_BAD)]
    for y, (lab, v, col) in zip((1, 0), rows):
        ax.barh(y, v[0], 0.55, color=col, xerr=[[v[0] - v[1]], [v[2] - v[0]]], capsize=3,
                error_kw={"lw": 1, "ecolor": "#333"})
        ax.text(v[1] - 1, y, f"{v[0]:.1f}%", va="center", ha="right", color="white", fontweight="bold")
        ax.text(1, y + 0.4, lab, va="center", fontsize=9.5)
    ax.axvline(75.2, color="k", ls=":", lw=1.4)
    ax.text(75.9, 1.62, "LIBERO-Plus reported: 75.2%", ha="left", fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.45, 1.8)
    ax.set_xlabel("OpenVLA-OFT success on all 379 LIBERO-Plus Goal 'sensor noise' tasks (%)")
    ax.spines["left"].set_visible(False)
    fig.suptitle("A file-name suffix in the instruction costs 13 points", fontsize=12.5, fontweight="bold",
                 x=0.02, ha="left")
    fig.text(0.02, 0.83, 'Example instruction the policy receives: "open the middle drawer of the cabinet view 0 0 100 0 0 initstate 0 noise 3"\n'
             'Same 379 tasks, paired: exact McNemar p = 4×10⁻⁷', fontsize=8.5, color="#444")
    fig.subplots_adjust(top=0.72, bottom=0.2, left=0.03, right=0.98)
    save(fig, "fig2_suffix")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    print("wrote", os.path.join(OUT, name + ".{png,pdf}"))


if __name__ == "__main__":
    fig1()
    fig2()
