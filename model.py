"""Baseline credit model: logistic regression trained on months 1-12, then frozen."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from data_generator import FEATURES


def _engineer(X: pd.DataFrame) -> np.ndarray:
    X = X[FEATURES].to_numpy(dtype=float).copy()
    X[:, 0] = np.log(X[:, 0])  # income enters the model on a log scale
    return X


def train_model(train_df: pd.DataFrame):
    model = make_pipeline(FunctionTransformer(_engineer), StandardScaler(), LogisticRegression(C=1.0))
    model.fit(train_df[FEATURES], train_df["default"])
    return model


def score(model, df: pd.DataFrame) -> np.ndarray:
    """Predicted probability of default."""
    return model.predict_proba(df[FEATURES])[:, 1]
