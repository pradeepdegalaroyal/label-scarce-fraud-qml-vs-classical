"""Tuned classical baselines (logistic regression, RBF-kernel SVM, random forest)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .qkernel import C_GRID

GAMMA_GRID = (0.05, 0.15, 0.5, 1.5)


def _cv(y: np.ndarray, seed: int) -> StratifiedKFold:
    folds = max(2, min(3, int(y.sum())))
    return StratifiedKFold(folds, shuffle=True, random_state=seed)


def fit_predict_classical(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Return decision scores on the test set for each tuned classical model."""
    cv = _cv(y_train, seed)
    out: dict[str, np.ndarray] = {}

    logreg = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
        {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
        scoring="average_precision", cv=cv,
    ).fit(x_train, y_train)
    out["LogReg"] = logreg.decision_function(x_test)

    svm = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
        {"svc__C": list(C_GRID), "svc__gamma": list(GAMMA_GRID)},
        scoring="average_precision", cv=cv,
    ).fit(x_train, y_train)
    out["RBF-SVM"] = svm.decision_function(x_test)

    forest = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=4
    ).fit(x_train, y_train)
    out["RandomForest"] = forest.predict_proba(x_test)[:, 1]
    return out


def ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    return float(average_precision_score(y_true, scores))
