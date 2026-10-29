"""Visual drift demonstrations for the presentation.

Shows, month by month, what each type of drift does to the data and to the
frozen model, using the same data generator and model as the experiment.

    from demo_plots import simulate, draw, animate
    sim = simulate("concept")          # "data", "label", "concept", "none", ...
    draw(sim, month=30)                # one snapshot
    animate(sim, "concept.gif")        # month-by-month animation

Each picture has four panels:
  A  Inputs:        one feature, training months vs the current month (+ PSI)
  B  Relationship:  default rate by credit-score band, before vs now,
                    and what the frozen model predicts for each band
  C  Default rate:  actual vs predicted, month by month
  D  Ranking:       the model's AUC, month by month
Outcomes in C and D are only known 6 months later; that window is shaded.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.metrics import roc_auc_score

from data_generator import SCENARIOS, generate_history
from detectors import psi
from model import score, train_model

TRAIN_MONTHS, ONSET, LAST, LABEL_DELAY = 12, 19, 36, 6

# Short names used in the notebook -> scenario in data_generator.py
SCENARIO_KEYS = {
    "none": "No drift (control)",
    "data": "Covariate drift (key features)",
    "data_minor": "Covariate drift (minor feature)",
    "label": "Prior drift",
    "concept": "Concept drift (sudden)",
    "concept_incremental": "Concept drift (incremental)",
}
TITLES = {
    "none": "Reference: no drift",
    "data": "Data drift: incomes and credit scores fall",
    "data_minor": "Data drift on a minor feature: job tenure rises",
    "label": "Label drift: the default rate doubles",
    "concept": "Concept drift: the same applicant means different risk",
    "concept_incremental": "Concept drift, phased in over 9 months",
}
# Which input feature panel A shows for each scenario.
SHOWN_FEATURE = {"none": "credit_score", "data": "income", "data_minor": "years_employed",
                 "label": "credit_score", "concept": "credit_score", "concept_incremental": "credit_score"}
FEATURE_LABEL = {"income": "Income ($k)", "credit_score": "Credit score", "years_employed": "Years employed",
                 "dti": "Debt-to-income"}
FEATURE_RANGE = {"income": (0, 200), "credit_score": (450, 850), "years_employed": (0, 30), "dti": (0, 1)}

BLUE, ORANGE, GREY, INK, INK2, SHADE = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e", "#ecebe7"
BANDS = np.arange(540, 851, 30)  # credit-score bands for the relationship panel


def simulate(key="concept", seed=1, n=10_000):
    """Generate 36 months for one scenario, train the frozen model, and compute monthly stats."""
    sc = next(s for s in SCENARIOS if s.name == SCENARIO_KEYS[key])
    hist = generate_history(seed, sc, months=LAST, n=n)
    train = hist[hist.month <= TRAIN_MONTHS]
    model = train_model(train)
    hist["pred"] = score(model, hist)
    train = hist[hist.month <= TRAIN_MONTHS]
    feat = SHOWN_FEATURE[key]
    rows = []
    for m, g in hist.groupby("month"):
        rows.append({"month": m, "actual": g["default"].mean(), "predicted": g["pred"].mean(),
                     "auc": roc_auc_score(g["default"], g["pred"]),
                     "psi": psi(train[feat].to_numpy(), g[feat].to_numpy())})
    return {"key": key, "hist": hist, "train": train, "model": model, "feature": feat,
            "monthly": pd.DataFrame(rows).set_index("month")}


def _risk_curve(df, col):
    band = pd.cut(df["credit_score"], BANDS)
    return df.groupby(band, observed=True)[col].mean()


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#d9d7d0")
    ax.spines["bottom"].set_color("#d9d7d0")
    ax.tick_params(colors=INK2, labelsize=11)
    ax.grid(color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)


def _feature_values(df, feat):
    return df[feat] / 1000 if feat == "income" else df[feat]


def draw(sim, month, fig=None):
    """Draw the four-panel picture for one month. Returns the figure."""
    if fig is None:
        fig = plt.figure(figsize=(12, 6.6), facecolor="white")
    fig.clf()
    axA, axB, axC, axD = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.1, hspace=0.55, wspace=0.2)
    hist, train, mon, feat = sim["hist"], sim["train"], sim["monthly"], sim["feature"]
    cur = hist[hist.month == month]
    drifting = month >= ONSET
    col = ORANGE if drifting else BLUE

    fig.suptitle(TITLES[sim["key"]], x=0.07, ha="left", fontsize=19, fontweight="bold", color=INK)
    if sim["key"] == "none":
        drifting = False  # the control never drifts
        col = BLUE
    status = f"Month {month}  ·  " + ("no drift in this scenario: what normal looks like" if sim["key"] == "none" else
                                      "drift has started (month 19)" if drifting else
                                      "training period" if month <= TRAIN_MONTHS else "no drift yet")
    fig.text(0.07, 0.895, status, fontsize=14, color=col if drifting else INK2, fontweight="bold")

    # A: inputs
    lo, hi = FEATURE_RANGE[feat]
    bins = np.linspace(lo, hi, 41)
    axA.hist(_feature_values(train, feat), bins=bins, density=True, color="#cfcdc6", label="Training (months 1–12)")
    axA.hist(_feature_values(cur, feat), bins=bins, density=True, histtype="step", lw=2.2, color=col,
             label=f"Month {month}")
    p = mon.loc[month, "psi"]
    axA.set_title(f"A · Inputs: {FEATURE_LABEL[feat].lower()}   PSI = {p:.2f}" + ("  ⚠" if p > 0.25 else ""),
                  loc="left", fontsize=14, color=INK, fontweight="bold")
    axA.set_yticks([])
    axA.set_ylim(0, axA.get_ylim()[1] * 1.45)
    axA.legend(frameon=False, fontsize=11, loc="upper right")

    # B: relationship P(y | X), by credit-score band
    mids = (BANDS[:-1] + BANDS[1:]) / 2
    base = _risk_curve(train, "default").to_numpy() * 100
    now = _risk_curve(cur, "default").reindex(_risk_curve(train, "default").index).to_numpy() * 100
    predn = _risk_curve(cur, "pred").reindex(_risk_curve(train, "default").index).to_numpy() * 100
    axB.plot(mids, base, color=GREY, lw=2, label="Actual, training")
    axB.plot(mids, now, color=col, lw=2.4, marker="o", ms=4, label=f"Actual, month {month}")
    axB.plot(mids, predn, color=BLUE, lw=1.6, ls="--", label="Model predicts")
    axB.set_ylim(0, 55)
    axB.set_title("B · Relationship: default rate by credit score", loc="left", fontsize=14, fontweight="bold", color=INK)
    axB.set_ylabel("Default rate (%)", fontsize=11, color=INK2)
    axB.legend(frameon=False, fontsize=11, loc="upper right")

    # C and D: outcomes over time (known only LABEL_DELAY months later)
    seen = mon.loc[TRAIN_MONTHS + 1: month]
    for ax in (axC, axD):
        ax.set_xlim(TRAIN_MONTHS + 1, LAST)
        if sim["key"] != "none":
            ax.axvline(ONSET, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.axvspan(max(month - LABEL_DELAY, TRAIN_MONTHS + 1) + 0.5, month + 0.5, color=SHADE, lw=0)
    axC.text(month - LABEL_DELAY / 2 + 0.5, 17.3, "not yet\nknown", ha="center", fontsize=10, color=INK2) \
        if month - LABEL_DELAY >= TRAIN_MONTHS else None
    axC.plot(seen.index, seen["actual"] * 100, color=ORANGE, lw=2.2, label="Actual")
    axC.plot(seen.index, seen["predicted"] * 100, color=BLUE, lw=2.2, label="Model predicts")
    axC.set_ylim(0, 20)
    a, pr = mon.loc[month, "actual"] * 100, mon.loc[month, "predicted"] * 100
    axC.set_title(f"C · Default rate   actual {a:.1f}% vs predicted {pr:.1f}%", loc="left", fontsize=14,
                  fontweight="bold", color=INK)
    axC.set_ylabel("%", fontsize=11, color=INK2)
    axC.set_xlabel("Month", fontsize=11, color=INK2)
    axC.legend(frameon=False, fontsize=11, loc="lower left", ncol=2)

    ref = roc_auc_score(train["default"], train["pred"])
    axD.axhline(ref, color=GREY, lw=1.2)
    axD.text(LAST - 0.3, ref + 0.008, f"training AUC {ref:.2f}", fontsize=10, color=INK2, ha="right")
    axD.plot(seen.index, seen["auc"], color=ORANGE if mon.loc[month, "auc"] < ref - 0.04 else BLUE, lw=2.2)
    axD.set_ylim(0.65, 0.9)
    axD.set_title(f"D · Ranking: model AUC   {mon.loc[month, 'auc']:.2f}", loc="left", fontsize=14,
                  fontweight="bold", color=INK)
    axD.set_xlabel("Month", fontsize=11, color=INK2)
    for ax in (axA, axB, axC, axD):
        _style(ax)
    return fig


def animate(sim, path=None, months=None, fps=2, hold=8, dpi=90):
    """Animate months 13-36. Saves a GIF if path is given; returns the animation."""
    months = list(months or range(TRAIN_MONTHS + 1, LAST + 1))
    frames = months + [months[-1]] * hold
    fig = plt.figure(figsize=(12, 6.6), facecolor="white")
    anim = FuncAnimation(fig, lambda m: draw(sim, m, fig), frames=frames, interval=1000 / fps)
    if path:
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return anim


def summary_table(keys=("data", "label", "concept"), seed=1):
    """Before (month 18) vs after (month 30) for the simple drift measures."""
    rows = []
    for k in keys:
        mon = simulate(k, seed)["monthly"]
        rows.append({"Drift": TITLES[k].split(":")[0], "Input PSI (shown feature)": round(mon.loc[30, "psi"], 2),
                     "Actual default rate": f"{mon.loc[30, 'actual']:.1%}",
                     "Predicted default rate": f"{mon.loc[30, 'predicted']:.1%}",
                     "AUC before → after": f"{mon.loc[18, 'auc']:.2f} → {mon.loc[30, 'auc']:.2f}"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Writes figures/demo/<scenario>.gif and before/after PNGs for the slides.
    from pathlib import Path
    out = Path("figures/demo")
    out.mkdir(parents=True, exist_ok=True)
    draw(simulate("none"), 24).savefig(out / "none_reference.png", dpi=150)
    plt.close("all")
    for k in ("data", "label", "concept"):
        s = simulate(k)
        for m, tag in ((18, "before"), (30, "after")):
            draw(s, m).savefig(out / f"{k}_{tag}.png", dpi=150)
            plt.close("all")
        animate(s, out / f"{k}.gif")
        print("saved", k)
    print(summary_table().to_string(index=False))
