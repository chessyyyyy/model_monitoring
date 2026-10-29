"""Drift detectors, written from scratch so every step can be explained.

Layer 1 - inputs (no labels):      PSI, KS test, domain classifier
Layer 2 - predictions (no labels): PSI on model scores
Layer 3 - outcomes (delayed):      calibration test, AUC, Page-Hinkley, ADWIN
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ---------------------------------------------------------------- Layer 1 & 2


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index.

    Bins come from deciles of the reference sample, so each bin holds ~10% of
    reference data. PSI = sum over bins of (a - e) * ln(a / e), where e and a
    are the reference and current shares in each bin.
    Rule of thumb: < 0.1 stable, 0.1-0.25 monitor, > 0.25 investigate.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(reference, edges)[0] / len(reference)
    a = np.histogram(current, edges)[0] / len(current)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_pvalue(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov test: largest gap between the two CDFs."""
    return float(stats.ks_2samp(reference, current).pvalue)


def domain_classifier_auc(reference: np.ndarray, current: np.ndarray, seed: int = 0) -> float:
    """Can a classifier tell this month's applicants from training applicants?

    Label reference rows 0 and current rows 1, fit a logistic regression with
    squared and interaction terms (so it sees shifts in means, spreads and
    correlations), and score it out-of-fold. AUC near 0.5 means the two
    samples are indistinguishable, i.e. no multivariate drift.
    """
    X = np.vstack([reference, current])
    y = np.r_[np.zeros(len(reference)), np.ones(len(current))]
    clf = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                        LogisticRegression(max_iter=500))
    p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of an AUC estimate."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(var))


# ------------------------------------------------------------------- Layer 3


def calibration_z(y: np.ndarray, p: np.ndarray) -> float:
    """Observed minus expected defaults, in standard errors.

    Under a well-calibrated model the number of defaults is a sum of
    Bernoulli(p_i), with mean sum(p) and variance sum(p(1-p)).
    """
    return float((y.sum() - p.sum()) / np.sqrt(np.sum(p * (1 - p))))


class PageHinkley:
    """Page-Hinkley test for an increase in the mean of a stream.

    Tracks the cumulative sum of (x - running mean - delta). An alarm fires
    when that sum rises more than `threshold` above its historical minimum.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0):
        self.delta, self.threshold = delta, threshold
        self.n, self.mean, self.cum, self.min_cum = 0, 0.0, 0.0, 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        return self.cum - self.min_cum > self.threshold

    def update_batch(self, xs) -> bool:
        alarm = False
        for x in xs:
            alarm |= self.update(float(x))
        return alarm


class ADWIN:
    """ADaptive WINdowing (Bifet & Gavalda, 2007), checked at monthly granularity.

    Keeps a window of recent monthly batches. After each new batch it tries
    every split into an older and a newer sub-window; if their means differ by
    more than a confidence bound, the older part is dropped and an alarm
    fires. We use the variance-aware bound from the ADWIN paper, which is far
    tighter than the plain Hoeffding bound when values (here per-loan Brier
    scores in [0, 1]) are mostly small.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.sums, self.sqs, self.counts = [], [], []

    def update_batch(self, xs) -> bool:
        xs = np.asarray(xs, dtype=float)
        self.sums.append(xs.sum())
        self.sqs.append((xs**2).sum())
        self.counts.append(len(xs))
        alarm = False
        while len(self.counts) > 1:
            total_n = sum(self.counts)
            mean = sum(self.sums) / total_n
            var = sum(self.sqs) / total_n - mean**2
            log_term = np.log(2 * np.log(total_n) / self.delta)
            cut = False
            for k in range(1, len(self.counts)):
                n0, n1 = sum(self.counts[:k]), sum(self.counts[k:])
                mu0, mu1 = sum(self.sums[:k]) / n0, sum(self.sums[k:]) / n1
                m = 1.0 / (1.0 / n0 + 1.0 / n1)
                eps = np.sqrt(2 * var * log_term / m) + 2 * log_term / (3 * m)
                if abs(mu0 - mu1) > eps:
                    cut = True
                    break
            if not cut:
                break
            alarm = True
            self.sums.pop(0)
            self.sqs.pop(0)
            self.counts.pop(0)
        return alarm
