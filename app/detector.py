"""Детектор: логистическая модель над признаками из features.py.

Модель хранится в JSON (средние, масштабы, веса), поэтому инференс — это скалярное произведение,
а вклад каждого признака = вес × стандартизированное значение. Эти вклады и показываются
пользователю как «ключевые предикторы».
"""
import datetime as dt
import json
import math
import os

import numpy as np

from . import config
from .features import FEATURES, describe, ladder

MATURE_FEATURES = {"pub_log_total", "pub_age", "news_saturated", "mainstream_share", "gh_log_total", "wiki_age",
                   "maturity_share"}
HYPE_FEATURES = {"pr_share"}

STAGE_LABELS = {1: "Концепция / исследование", 2: "Прототип / PoC", 3: "Пилот", 4: "Раннее внедрение",
                5: "Массовое внедрение"}
TREND_LABELS = {1: "Стабильный", 2: "Растёт", 3: "Растёт быстро"}
GATE_THRESHOLD = 0.60
GATE_C = 0.05
GATE_SIGNS = {
    "wiki_age": -1, "news_saturated": -1, "pub_log_total": -1, "mainstream_share": -1,
    "funding_share": 1, "pub_growth_p": 1, "news_growth_p": 1, "news_burst": 1,
}
GATE_SIGN_MIN_ABS = 0.001

# Априорные веса: знаки заданы вручную из методологии. Используются, пока модель не обучена,
# и как бейзлайн в отчёте о метриках. (среднее, масштаб, вес)
PRIOR = {
    "pub_log_total": (6.0, 2.5, -0.8), "pub_growth_p": (0.5, 0.3, 0.6), "pub_recent_share": (0.4, 0.2, 0.5),
    "pub_age": (8.0, 6.0, -0.5), "news_log_recent": (3.0, 1.2, 0.2), "news_growth_p": (0.5, 0.3, 0.5),
    "news_saturated": (0.3, 0.45, -0.7), "news_burst": (0.15, 0.3, 0.4), "funding_share": (0.1, 0.1, 0.6), "pr_share": (0.2, 0.2, -0.4),
    "mainstream_share": (0.1, 0.1, -0.5), "domain_diversity": (0.7, 0.2, 0.3), "gh_log_recent": (2.0, 1.5, 0.2),
    "gh_log_total": (4.0, 2.5, -0.4), "wiki_age": (5.0, 6.0, -0.8), "maturity_share": (0.03, 0.05, -0.4),
}


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))


def prior_model() -> dict:
    return {"source": "prior", "features": FEATURES, "trained_at": None,
            "gate_threshold": 0.5,
            "gate": {"mean": [PRIOR[f][0] for f in FEATURES], "scale": [PRIOR[f][1] for f in FEATURES],
                     "coef": [PRIOR[f][2] for f in FEATURES], "intercept": 0.0},
            "stage": None, "trend": None, "metrics": None}


_MODEL = {"value": None, "mtime": None}


def load_model() -> dict:
    path = config.MODEL_PATH
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        if _MODEL["mtime"] != mtime:
            with open(path, encoding="utf-8") as fh:
                _MODEL["value"], _MODEL["mtime"] = json.load(fh), mtime
        return _MODEL["value"]
    return prior_model()


def _vector(f: dict, names) -> np.ndarray:
    return np.array([float(f.get(n, 0.0)) for n in names])


def gate(f: dict, model: dict | None = None) -> tuple[float, dict]:
    model = model or load_model()
    g = model["gate"]
    names = model["features"]
    z = (_vector(f, names) - np.array(g["mean"])) / np.where(np.array(g["scale"]) == 0, 1, np.array(g["scale"]))
    contrib = np.array(g["coef"]) * z
    p = _sigmoid(float(contrib.sum() + g["intercept"]))
    return p, {n: float(c) for n, c in zip(names, contrib)}


def is_weak_signal(p: float, model: dict | None = None) -> bool:
    threshold = model.get("gate_threshold", 0.5) if model is not None else GATE_THRESHOLD
    return p >= threshold


def _softmax_predict(block: dict, f: dict, names) -> int:
    x = (_vector(f, names) - np.array(block["mean"])) / np.where(np.array(block["scale"]) == 0, 1,
                                                                 np.array(block["scale"]))
    coef = np.array(block["coef"])
    inter = np.array(block["intercept"])
    if coef.ndim == 1 or coef.shape[0] == 1:  # бинарная модель
        p = _sigmoid(float(np.ravel(coef) @ x + float(np.ravel(inter)[0])))
        return block["classes"][1] if p >= 0.5 else block["classes"][0]
    scores = coef @ x + inter
    return int(block["classes"][int(np.argmax(scores))])


def stage_trend(f: dict, model: dict | None = None) -> tuple[int, int]:
    model = model or load_model()
    names = model["features"]
    lad = ladder(f)
    if model.get("stage"):
        stage = _softmax_predict(model["stage"], f, names)
    else:  # правило по лестнице зрелости
        stage = (5 if lad["encyclopedia"] and lad["mainstream"] else 4 if lad["money"] and lad["mainstream"]
                 else 3 if lad["money"] else 2 if lad["code"] else 1)
    if model.get("trend"):
        trend = _softmax_predict(model["trend"], f, names)
    else:
        g = max(f["news_growth_p"], f["pub_growth_p"])
        trend = 3 if g >= 0.8 else 2 if g >= 0.5 else 1
    return int(stage), int(trend)


def explain(f: dict, contrib: dict, top: int = 4) -> dict:
    items = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    pros = [{"feature": k, "contribution": round(v, 3), "text": describe(k, f[k])} for k, v in items if v > 0.05][:top]
    cons = [{"feature": k, "contribution": round(v, 3), "text": describe(k, f[k])}
            for k, v in reversed(items) if v < -0.05][:top]
    return {"pros": pros, "cons": cons}


def rejection(f: dict, contrib: dict, evidence_docs: int) -> tuple[str, str]:
    """Категория и причина исключения кандидата."""
    if evidence_docs < 3 and f["pub_log_total"] < math.log1p(3) and f["gh_log_total"] < math.log1p(1):
        return "шум", "недостаточно независимых подтверждений в открытых источниках"
    neg = {k: v for k, v in contrib.items() if v < 0}
    mature = -sum(v for k, v in neg.items() if k in MATURE_FEATURES)
    hype = -sum(v for k, v in neg.items() if k in HYPE_FEATURES)
    top = sorted(neg.items(), key=lambda kv: kv[1])[:2]
    reason = "; ".join(describe(k, f[k]) for k, _ in top) or "низкая итоговая уверенность"
    if hype > mature and hype > 0:
        return "хайп", reason
    if mature > 0:
        return "зрелая технология", reason
    return "слабые подтверждения", reason


# ---------------------------------------------------------------- обучение
def _fit_block(X, y, names, multi=False, C=1.0, sign_constraints=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    if sign_constraints:
        from scipy.optimize import minimize
        from scipy.special import expit

        if multi:
            raise ValueError("Sign constraints apply only to the binary gate")
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("The constrained gate requires two classes")
        x = sc.transform(X)
        target = (np.asarray(y) == classes[1]).astype(float)
        n = len(target)
        sample_weight = np.where(target == 1, n / (2 * counts[1]), n / (2 * counts[0]))
        regularization = 1 / (C * n)

        def objective(params):
            coef, intercept = params[:-1], params[-1]
            logit = x @ coef + intercept
            error = sample_weight * (expit(logit) - target)
            loss = (np.dot(sample_weight, np.logaddexp(0, logit) - target * logit) / n
                    + regularization * np.dot(coef, coef) / 2)
            gradient = np.r_[x.T @ error / n + regularization * coef, error.sum() / n]
            return loss, gradient

        bounds = []
        for name in names:
            sign = GATE_SIGNS.get(name)
            bounds.append((None, -GATE_SIGN_MIN_ABS) if sign == -1 else
                          (GATE_SIGN_MIN_ABS, None) if sign == 1 else (None, None))
        result = minimize(objective, np.zeros(len(names) + 1), jac=True, method="L-BFGS-B",
                          bounds=bounds + [(None, None)], options={"maxiter": 5000, "ftol": 1e-12})
        if not result.success:
            raise RuntimeError(f"Constrained gate fit failed: {result.message}")
        return {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": result.x[:-1].tolist(),
                "intercept": float(result.x[-1]), "classes": [int(c) for c in classes]}
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(sc.transform(X), y)
    return {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_.tolist() if multi or
            len(clf.classes_) > 2 else clf.coef_[0].tolist(), "intercept": clf.intercept_.tolist() if multi or
            len(clf.classes_) > 2 else float(clf.intercept_[0]), "classes": [int(c) for c in clf.classes_]}


def train(rows: list[dict], names=None, C: float = GATE_C) -> dict:
    """rows: {features..., label (1 слабый / 0 нет), area, stage (1–5), trend (1–3)}.
    Кросс-валидация: по темам (leave-one-theme-out), чтобы близкие технологии одной темы не
    попадали одновременно в обучение и проверку."""
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                                 roc_auc_score)
    names = names or FEATURES
    X = np.array([[float(r[n]) for n in names] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    areas = np.array([r["area"] for r in rows])
    stages = np.array([int(r["stage"]) for r in rows])
    trends = np.array([3 if int(r["trend"]) == 3 else 2 for r in rows])

    oof_p = np.zeros(len(rows))
    oof_stage = np.zeros(len(rows), dtype=int)
    oof_trend = np.zeros(len(rows), dtype=int)
    per_area = {}
    for area in sorted(set(areas)):
        test = areas == area
        train_idx = ~test
        g = _fit_block(X[train_idx], y[train_idx], names, C=C, sign_constraints=True)
        m = {"features": names, "gate": g}
        oof_p[test] = [gate(dict(zip(names, x)), m)[0] for x in X[test]]
        st = _fit_block(X[train_idx], stages[train_idx], names, multi=True)
        pos = train_idx & (y == 1)
        tr = _fit_block(X[pos], trends[pos], names)
        for i in np.where(test)[0]:
            f = dict(zip(names, X[i]))
            oof_stage[i] = _softmax_predict(st, f, names)
            oof_trend[i] = _softmax_predict(tr, f, names)
        pred = (oof_p[test] >= GATE_THRESHOLD).astype(int)
        per_area[area] = {"n": int(test.sum()), "accuracy": float(accuracy_score(y[test], pred)),
                          "recall_weak": float(recall_score(y[test], pred, zero_division=0))}

    pred = (oof_p >= GATE_THRESHOLD).astype(int)
    pos = y == 1
    prior = prior_model()
    prior_pred = np.array([gate(r, prior)[0] >= 0.5 for r in rows]).astype(int)
    score_true = stages[pos] + np.array([int(r["trend"]) for r in rows])[pos]
    score_pred = np.minimum(oof_stage[pos], 4) + oof_trend[pos]
    metrics = {
        "n": int(len(rows)), "n_weak": int(pos.sum()), "n_not_weak": int((~pos).sum()),
        "cv": "leave-one-theme-out",
        "gate": {"accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred)),
                 "recall": float(recall_score(y, pred)), "f1": float(f1_score(y, pred)),
                 "roc_auc": float(roc_auc_score(y, oof_p)) if len(set(y)) > 1 else None,
                 "confusion": confusion_matrix(y, pred).tolist()},
        "prior_baseline": {"accuracy": float(accuracy_score(y, prior_pred)), "f1": float(f1_score(y, prior_pred))},
        "stage_exact": float(np.mean(oof_stage[pos] == stages[pos])),
        "stage_within1": float(np.mean(np.abs(oof_stage[pos] - stages[pos]) <= 1)),
        "trend_exact": float(np.mean(oof_trend[pos] == trends[pos])),
        "score_exact": float(np.mean(score_pred == score_true)),
        "score_within1": float(np.mean(np.abs(score_pred - score_true) <= 1)),
        "per_area": per_area,
    }
    final_pos = y == 1
    return {
        "source": "trained", "features": names, "trained_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "gate_threshold": GATE_THRESHOLD, "gate_C": C,
        "gate_sign_constraints": GATE_SIGNS,
        "gate": _fit_block(X, y, names, C=C, sign_constraints=True),
        "stage": _fit_block(X, stages, names, multi=True),
        "trend": _fit_block(X[final_pos], trends[final_pos], names),
        "metrics": metrics,
    }
