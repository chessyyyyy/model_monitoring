"""Synthetic loan-default data with controllable drift.

Each month we draw a batch of loan applicants with four features and a
default outcome. The true default probability follows a logistic model
whose coefficients we control, so we know exactly when and how the
world changes (ground truth for evaluating detectors).

Drift types, in terms of the joint distribution P(X, y):
    covariate  -> P(X) changes,      P(y | X) fixed
    prior      -> P(y) changes,      P(X | y) fixed
    concept    -> P(y | X) changes,  P(X) fixed
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURES = ["income", "dti", "credit_score", "years_employed"]

# Fixed standardisation constants (roughly the training-population moments),
# so the true risk function is the same formula in every month.
_CENTER = {"log_income": np.log(60_000), "dti": 0.30, "credit_score": 690.0, "years_employed": 6.0}
_SCALE = {"log_income": 0.45, "dti": 0.11, "credit_score": 60.0, "years_employed": 4.2}

# Baseline true coefficients on standardised features (~8% default rate).
BASE_BETA = {"intercept": -3.24, "income": -0.5, "dti": 0.8, "credit_score": -0.9, "years_employed": -0.3}

# Concept drift: income no longer protects, credit history barely matters and
# payment burden (DTI) matters more. The intercept is re-solved so the overall
# default rate stays at ~8%: only the *ranking* of risk changes.
CONCEPT_BETA = {"intercept": -3.11, "income": 0.0, "dti": 1.0, "credit_score": -0.1}


@dataclass
class Scenario:
    """Describes one drift experiment.

    kind:   "none", "covariate", "prior", "concept"
    onset:  first month (1-indexed) affected by the drift
    ramp:   months over which the drift phases in (1 = sudden)
    """
    name: str
    kind: str = "none"
    onset: int = 19
    ramp: int = 1
    params: dict = field(default_factory=dict)

    def strength(self, month: int) -> float:
        """Fraction of the full drift applied in a given month (0 to 1)."""
        if self.kind == "none" or month < self.onset:
            return 0.0
        return min(1.0, (month - self.onset + 1) / self.ramp)


SCENARIOS = [
    Scenario("No drift (control)", "none"),
    # Incomes fall 30% and credit scores drop 30 points. The risk formula is
    # unchanged, so a correctly specified model stays right.
    Scenario("Covariate drift (key features)", "covariate",
             params={"income_mult": 0.70, "score_shift": -30.0}),
    # Applicants have much longer job tenure. A real input shift, but on a
    # feature the model barely relies on.
    Scenario("Covariate drift (minor feature)", "covariate", params={"tenure_mult": 1.8}),
    # Share of defaulters doubles (8% -> 16%); defaulters and non-defaulters
    # each still look the way they always did.
    Scenario("Prior drift", "prior", params={"target_rate": 0.16}),
    # Same applicants, different risk (see CONCEPT_BETA).
    Scenario("Concept drift (sudden)", "concept", params=CONCEPT_BETA),
    Scenario("Concept drift (incremental)", "concept", ramp=9, params=CONCEPT_BETA),
]


def _draw_features(rng, n, income_mult=1.0, score_shift=0.0, tenure_mult=1.0):
    income = rng.lognormal(mean=np.log(60_000), sigma=0.45, size=n) * income_mult
    dti = rng.beta(3.0, 7.0, size=n)  # mean 0.30
    credit_score = np.clip(rng.normal(690 + score_shift, 60, size=n), 300, 850)
    years_employed = rng.gamma(shape=2.0, scale=3.0, size=n) * tenure_mult  # mean 6
    return pd.DataFrame({"income": income, "dti": dti, "credit_score": credit_score,
                         "years_employed": years_employed})


def standardise(X: pd.DataFrame) -> np.ndarray:
    """Map raw features to the standardised scale used by the true risk formula."""
    return np.column_stack([
        (np.log(X["income"]) - _CENTER["log_income"]) / _SCALE["log_income"],
        (X["dti"] - _CENTER["dti"]) / _SCALE["dti"],
        (X["credit_score"] - _CENTER["credit_score"]) / _SCALE["credit_score"],
        (X["years_employed"] - _CENTER["years_employed"]) / _SCALE["years_employed"],
    ])


def true_default_prob(X: pd.DataFrame, beta: dict) -> np.ndarray:
    Z = standardise(X)
    logit = beta["intercept"] + Z @ np.array([beta[f] for f in FEATURES])
    return 1.0 / (1.0 + np.exp(-logit))


def generate_month(rng, scenario: Scenario, month: int, n: int = 2000) -> pd.DataFrame:
    """Draw one month of applicants under the given scenario."""
    s = scenario.strength(month)
    beta = dict(BASE_BETA)
    shift = {"income_mult": 1.0, "score_shift": 0.0, "tenure_mult": 1.0}

    if scenario.kind == "covariate":
        for k, v in scenario.params.items():
            shift[k] += s * (v - shift[k])
    elif scenario.kind == "concept":
        for k, v in scenario.params.items():
            beta[k] = BASE_BETA[k] + s * (v - BASE_BETA[k])

    if scenario.kind == "prior" and s > 0:
        # Resample by class: draw a large pool from the unchanged world, then
        # pick defaulters and non-defaulters at the new mix. This changes P(y)
        # while keeping P(X | y) exactly as before.
        pool = _draw_features(rng, n * 12)
        p = true_default_prob(pool, BASE_BETA)
        y = rng.random(len(pool)) < p
        base_rate = y.mean()
        rate = base_rate + s * (scenario.params["target_rate"] - base_rate)
        n_def = rng.binomial(n, rate)
        idx = np.concatenate([
            rng.choice(np.flatnonzero(y), n_def, replace=False),
            rng.choice(np.flatnonzero(~y), n - n_def, replace=False),
        ])
        rng.shuffle(idx)
        df = pool.iloc[idx].reset_index(drop=True)
        df["default"] = y[idx].astype(int)
        df["true_pd"] = p[idx]
    else:
        df = _draw_features(rng, n, **shift)
        df["true_pd"] = true_default_prob(df, beta)
        df["default"] = (rng.random(n) < df["true_pd"]).astype(int)

    df["month"] = month
    return df


def generate_history(seed: int, scenario: Scenario, months: int = 36, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.concat([generate_month(rng, scenario, m, n) for m in range(1, months + 1)],
                     ignore_index=True)
