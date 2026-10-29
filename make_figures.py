"""Turn results/metrics.csv and results/summary.csv into the charts used in the slides.

Usage:  python make_figures.py      (run after run_experiment.py)
Writes: figures/*.png  (16:9-friendly, 200 dpi)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from run_experiment import DETECTORS, LABEL_DELAY, PSI_ALARM, TRAIN_MONTHS

ONSET = 19
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True,
})

OUT = Path("figures")
SLIDES = OUT / "slides"  # untitled variants for the deck, where the slide title says it


def _save(fig, name, title_ax=None):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    if title_ax is not None:
        title_ax.set_title("")
    fig.savefig(SLIDES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _band(ax, df, col, color, label, x="month", lw=2.2):
    g = df.groupby(x)[col]
    med, lo, hi = g.median(), g.quantile(0.1), g.quantile(0.9)
    # Steps: each month's value holds for that whole month, so a change at
    # month 19 is drawn as a jump at 19 rather than a slope from 18.
    ax.fill_between(med.index, lo, hi, color=color, alpha=0.15, linewidth=0, step="post")
    ax.plot(med.index, med.values, color=color, lw=lw, drawstyle="steps-post")
    ax.annotate(label, (med.index[-1], med.values[-1]), xytext=(6, 0), textcoords="offset points",
                color=INK, fontsize=11, va="center", fontweight="bold")
    return med


def _onset(ax, text=True):
    ax.axvline(ONSET, color=INK2, lw=1.2, ls=(0, (4, 3)))
    if text:
        ax.text(ONSET + 0.3, 1.0, "drift starts", transform=ax.get_xaxis_transform(),
                color=INK2, fontsize=10, va="top")


def fig_covariate_vs_concept(m):
    """Slide 6: input monitoring sees covariate drift; only outcomes see concept drift."""
    cov = m[m.scenario == "Covariate drift (key features)"]
    con = m[m.scenario == "Concept drift (sudden)"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"hspace": 0.38})

    _band(a, cov, "max_feature_psi", BLUE, "Covariate drift")
    _band(a, con, "max_feature_psi", ORANGE, "Concept drift")
    a.axhline(PSI_ALARM, color=INK2, lw=1)
    a.text(13.2, PSI_ALARM + 0.02, f"alarm threshold ({PSI_ALARM})", color=INK2, fontsize=10)
    a.set_title("Input monitoring: largest feature PSI each month (available immediately)")
    a.set_ylabel("PSI")
    _onset(a)

    # Outcome metrics are plotted in the month the labels actually arrive.
    lab = m[m.labels_known].assign(arrival=lambda d: d.month + LABEL_DELAY)
    _band(b, lab[lab.scenario == cov.scenario.iloc[0]], "auc", BLUE, "Covariate drift", x="arrival")
    _band(b, lab[lab.scenario == con.scenario.iloc[0]], "auc", ORANGE, "Concept drift", x="arrival")
    b.axvspan(ONSET, ONSET + LABEL_DELAY, color=GRID, alpha=0.6, linewidth=0)
    b.text(ONSET + LABEL_DELAY / 2, 0.715, f"{LABEL_DELAY}-month label delay:\noutcomes arriving are\nfrom pre-drift loans",
           ha="center", color=INK2, fontsize=10)
    b.set_title("Outcome monitoring: model AUC, plotted when defaults become known")
    b.set_ylabel("AUC")
    b.set_ylim(0.70, 0.88)
    b.set_xlabel("Production month")
    _onset(b, text=False)
    for ax in (a, b):
        ax.set_xlim(TRAIN_MONTHS + 1, 39.5)
        ax.set_xticks(range(13, 37, 3))
    fig.text(0.01, 0.005, "Median across 30 simulations; shaded band = 10th-90th percentile.",
             color=INK2, fontsize=9)
    _save(fig, "covariate_vs_concept.png")


def fig_results_grid(s):
    """Slide 7: which detector caught which drift, and how many months after onset."""
    order = ["Covariate drift (key features)", "Covariate drift (minor feature)", "Prior drift",
             "Concept drift (sudden)", "Concept drift (incremental)", "No drift (control)"]
    heads = ["Covariate\n(key features)", "Covariate\n(minor feature)", "Prior\n(default rate x2)",
             "Concept\n(sudden)", "Concept\n(incremental)", "No drift\n(false alarms)"]
    dets = [d[0] for d in DETECTORS]
    rate = s.pivot(index="detector", columns="scenario", values="detection_rate").loc[dets, order]
    delay = s.pivot(index="detector", columns="scenario", values="median_delay_months").loc[dets, order]

    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.grid(False)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    vals = rate.to_numpy().copy()
    ax.imshow(vals[:, :-1], cmap=cmap, vmin=0, vmax=1, aspect="auto",
              extent=(-0.5, len(order) - 1.5, len(dets) - 0.5, -0.5))
    for i in range(len(dets)):
        for j in range(len(order)):
            r = vals[i, j]
            if j == len(order) - 1:  # control column: false alarm rate, no fill
                txt = f"{r:.0%}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                        color=ORANGE if r > 0.1 else INK2, fontweight="bold" if r > 0.1 else "normal")
                continue
            d = delay.iloc[i, j]
            txt = f"{r:.0%}" + (f"\n{d:.0f} mo" if r >= 0.5 and not np.isnan(d) else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    color="white" if r >= 0.55 else INK, fontweight="bold" if r >= 0.8 else "normal")
    # thin surface gaps between cells and a divider before the control column
    for k in range(len(dets) + 1):
        ax.axhline(k - 0.5, color=SURFACE, lw=3)
    for k in range(len(order)):
        ax.axvline(k - 0.5, color=SURFACE, lw=3)
    ax.axvline(len(order) - 1.5, color=INK2, lw=1)
    # layer separators
    for k in (3, 4):
        ax.axhline(k - 0.5, color=INK2, lw=1)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(len(dets) - 0.5, -0.5)
    ax.set_xticks(range(len(order)), heads, fontsize=10.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(dets)), dets, fontsize=11.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for y, name in [(1, "INPUTS"), (3, "PREDICTIONS"), (5.5, "OUTCOMES\n(labels 6 mo late)")]:
        ax.text(-0.235, y, name, transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=9.5, color=INK2, fontweight="bold", rotation=90)
    ax.set_title("Which monitor caught which drift?", pad=58)
    fig.text(0.01, 0.01, "Cell = share of 30 simulations with an alarm after drift began; "
             "'mo' = median months from drift start to first alarm.\n"
             "Last column = share of no-drift simulations with any (false) alarm over 24 months.",
             color=INK2, fontsize=9.5)
    _save(fig, "results_grid.png", ax)


def fig_label_delay(m):
    """Slides 1 and 4: the default rate doubles, the model keeps predicting the old rate."""
    p = m[m.scenario == "Prior drift"]
    fig, ax = plt.subplots(figsize=(11, 5))
    true = p.groupby("month").true_default_rate.median() * 100
    pred = p.groupby("month").mean_predicted_pd.median() * 100
    ax.plot(true.index, true.values, color=ORANGE, lw=2.4, drawstyle="steps-post")
    ax.plot(pred.index, pred.values, color=BLUE, lw=2.4, drawstyle="steps-post")
    ax.annotate("Actual default rate", (36, true.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontweight="bold", fontsize=11)
    ax.annotate("Model's predicted rate", (36, pred.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontweight="bold", fontsize=11)
    ax.axvspan(ONSET, ONSET + LABEL_DELAY, color=GRID, alpha=0.6, linewidth=0)
    ax.text(ONSET + LABEL_DELAY / 2, 5.0, "Losses are building,\nbut no defaults\nhave been observed yet",
            ha="center", color=INK2, fontsize=10)
    ax.axvline(ONSET + LABEL_DELAY, color=INK, lw=1.4)
    ax.text(ONSET + LABEL_DELAY + 0.3, 13.2, "First post-drift\noutcomes arrive\n(month 25)", fontsize=10, va="top")
    _onset(ax)
    ax.set_ylim(0, 18)
    ax.set_xlim(TRAIN_MONTHS + 1, 40.5)
    ax.set_xticks(range(13, 37, 3))
    ax.set_ylabel("Default rate (%)")
    ax.set_xlabel("Production month (by loan cohort)")
    ax.set_title("The world changed at month 19. Nobody can see it until month 25.")
    fig.text(0.01, -0.04, "Prior-drift simulation: share of defaulters doubles from 8% to 16%. "
             "Median of 30 simulations.", color=INK2, fontsize=9)
    _save(fig, "label_delay.png", ax)


def fig_drift_timing():
    """Slide 3: the four timing patterns (schematic)."""
    t = np.linspace(0, 10, 400)
    shapes = {
        "Sudden": np.where(t < 4, 0, 1.0),
        "Gradual": np.where((np.random.default_rng(3).random(400) < np.clip((t - 3) / 5, 0, 1)), 1.0, 0.0),
        "Incremental": np.clip((t - 3) / 5, 0, 1),
        "Recurring": (np.sin(t * 1.3) > 0).astype(float),
    }
    subs = {"Sudden": "e.g. COVID lockdown", "Gradual": "old and new regimes mix",
            "Incremental": "slow, steady change", "Recurring": "seasons, market regimes"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, (name, y) in zip(axes, shapes.items()):
        if name == "Gradual":
            ax.scatter(t[::4], y[::4] + np.random.default_rng(1).normal(0, 0.02, 100), s=10, color=BLUE)
        else:
            ax.plot(t, y, color=BLUE, lw=2.6)
        ax.set_title(name)
        ax.text(0, -0.42, subs[name], color=INK2, fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([0, 1], ["old\nworld", "new\nworld"])
        ax.set_ylim(-0.25, 1.25)
        ax.grid(False)
    axes[0].set_xlabel("")
    fig.text(0.5, -0.12, "time  →", ha="center", color=INK2)
    _save(fig, "drift_timing.png")


def main():
    SLIDES.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv("results/metrics.csv")
    s = pd.read_csv("results/summary.csv")
    fig_covariate_vs_concept(m)
    fig_results_grid(s)
    fig_label_delay(m)
    fig_drift_timing()
    print("Saved:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))), "+ figures/slides/")


if __name__ == "__main__":
    main()
