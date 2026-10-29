"""Run every drift scenario over many random seeds and log all monitoring signals.

Timeline
    months 1-12   training window (model is fit here, then frozen)
    months 13-36  production: a new batch of applicants is scored each month
    month 19      drift begins (all drift scenarios)
    label delay   6 months: defaults for cohort c are only known in month c + 6

Usage:  python run_experiment.py [--seeds 30] [--workers 4]
Writes: results/metrics.csv, results/summary.csv
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import detectors as D
from data_generator import FEATURES, SCENARIOS, generate_history
from model import score, train_model

TRAIN_MONTHS = 12
LAST_MONTH = 36
LABEL_DELAY = 6
BATCH = 2000

# Alarm rules. PSI uses the industry rule of thumb; statistical tests use
# 3-sigma or Bonferroni-corrected 1% levels; the two sequential detectors were
# tuned so the no-drift control rarely alarms (see README).
PSI_ALARM = 0.25
KS_ALPHA = 0.01 / len(FEATURES)
Z_ALARM = 3.0
PH_DELTA, PH_THRESHOLD = 0.002, 40.0
ADWIN_DELTA = 0.002

DETECTORS = [
    # name, layer, needs labels
    ("Feature PSI", "Inputs", False),
    ("Feature KS test", "Inputs", False),
    ("Domain classifier", "Inputs", False),
    ("Score PSI", "Predictions", False),
    ("Calibration test", "Outcomes", True),
    ("AUC drop", "Outcomes", True),
    ("Page-Hinkley", "Outcomes", True),
    ("ADWIN", "Outcomes", True),
]


def run_one(seed: int, scenario_idx: int) -> pd.DataFrame:
    scenario = SCENARIOS[scenario_idx]
    hist = generate_history(seed, scenario, months=LAST_MONTH, n=BATCH)
    train = hist[hist.month <= TRAIN_MONTHS]
    model = train_model(train)

    ref_X = train[FEATURES].to_numpy()
    ref_scores = score(model, train)
    ref_auc = roc_auc_score(train["default"], ref_scores)
    rng = np.random.default_rng(seed + 10_000)
    ref_sample = ref_X[rng.choice(len(ref_X), BATCH, replace=False)]

    ph = D.PageHinkley(PH_DELTA, PH_THRESHOLD)
    adwin = D.ADWIN(ADWIN_DELTA)
    rows = []

    for month in range(TRAIN_MONTHS + 1, LAST_MONTH + 1):
        cur = hist[hist.month == month]
        X = cur[FEATURES].to_numpy()
        p = score(model, cur)
        y = cur["default"].to_numpy()

        feat_psi = [D.psi(ref_X[:, j], X[:, j]) for j in range(len(FEATURES))]
        feat_ks = [D.ks_pvalue(ref_X[:, j], X[:, j]) for j in range(len(FEATURES))]
        dc_auc = D.domain_classifier_auc(ref_sample, X)
        dc_se = D.auc_standard_error(0.5, BATCH, BATCH)
        s_psi = D.psi(ref_scores, p)

        # Outcome metrics are computed for this cohort but only become
        # available LABEL_DELAY months later.
        auc = roc_auc_score(y, p)
        auc_se = D.auc_standard_error(ref_auc, int(y.sum()), int(len(y) - y.sum()))
        cal_z = D.calibration_z(y, p)
        labels_known = month + LABEL_DELAY <= LAST_MONTH
        brier = (p - y) ** 2
        ph_alarm = ph.update_batch(brier) if labels_known else False
        if ph_alarm:
            ph = D.PageHinkley(PH_DELTA, PH_THRESHOLD)  # reset after each alarm
        adwin_alarm = adwin.update_batch(brier) if labels_known else False

        rows.append({
            "seed": seed, "scenario": scenario.name, "month": month,
            "alarm_month_outcomes": month + LABEL_DELAY, "labels_known": labels_known,
            **{f"psi_{f}": v for f, v in zip(FEATURES, feat_psi)},
            "max_feature_psi": max(feat_psi), "min_ks_p": min(feat_ks),
            "domain_auc": dc_auc, "score_psi": s_psi,
            "auc": auc, "ref_auc": ref_auc, "auc_se": auc_se, "calibration_z": cal_z,
            "brier": brier.mean(),
            "true_default_rate": y.mean(), "mean_predicted_pd": p.mean(),
            "alarm_Feature PSI": max(feat_psi) > PSI_ALARM,
            "alarm_Feature KS test": min(feat_ks) < KS_ALPHA,
            "alarm_Domain classifier": dc_auc > 0.5 + Z_ALARM * dc_se,
            "alarm_Score PSI": s_psi > PSI_ALARM,
            "alarm_Calibration test": labels_known and abs(cal_z) > Z_ALARM,
            "alarm_AUC drop": labels_known and auc < ref_auc - Z_ALARM * auc_se,
            "alarm_Page-Hinkley": ph_alarm,
            "alarm_ADWIN": adwin_alarm,
        })
    return pd.DataFrame(rows)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Detection rate, median delay and false-alarm rate per detector and scenario."""
    onset = SCENARIOS[1].onset
    out = []
    for scen, (det_name, layer, needs_labels) in product([s.name for s in SCENARIOS], DETECTORS):
        sub = metrics[metrics.scenario == scen]
        col = f"alarm_{det_name}"
        seeds = sub.seed.unique()
        detected, delays, pre_onset = [], [], []
        for s in seeds:
            ss = sub[(sub.seed == s) & sub[col]]
            when = ss["alarm_month_outcomes"] if needs_labels else ss["month"]
            data_month = ss["month"]
            if scen.startswith("No drift"):
                detected.append(len(ss) > 0)
                continue
            pre_onset.append(bool((data_month < onset).any()))
            valid = when[data_month >= onset]
            detected.append(len(valid) > 0)
            if len(valid):
                delays.append(int(valid.min()) - onset)
        out.append({
            "scenario": scen, "detector": det_name, "layer": layer,
            "detection_rate": float(np.mean(detected)),
            "median_delay_months": float(np.median(delays)) if delays else np.nan,
            "pre_onset_false_alarm_rate": float(np.mean(pre_onset)) if pre_onset else np.nan,
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = [(seed, i) for i in range(len(SCENARIOS)) for seed in range(args.seeds)]
    with ProcessPoolExecutor(args.workers) as ex:
        frames = list(ex.map(run_one, *zip(*jobs)))
    metrics = pd.concat(frames, ignore_index=True)

    Path("results").mkdir(exist_ok=True)
    metrics.to_csv("results/metrics.csv", index=False)
    summary = summarise(metrics)
    summary.to_csv("results/summary.csv", index=False)
    pd.set_option("display.width", 160)
    print(summary.pivot(index="detector", columns="scenario", values="detection_rate").round(2))
    print(summary.pivot(index="detector", columns="scenario", values="median_delay_months"))


if __name__ == "__main__":
    main()
