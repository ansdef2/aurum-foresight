"""Эксперименты для отчёта и презентации. Один запуск даёт все цифры и рисунки.

    python -m scripts.experiments

Читает data/training/features.csv (и features_alt*.csv, если есть), пишет:
    docs/experiments.md     таблицы для слайдов и журнала экспериментов
    docs/experiments.json   те же числа в машиночитаемом виде
    docs/figures/*.png      рисунки для слайдов (если установлен matplotlib)

Все оценки — «вне выборки»: leave-one-theme-out (модель не видела тему, на которой её проверяют).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from app import detector
from app.features import FEATURES

IN = "data/training/features.csv"
OUT_MD, OUT_JSON, FIG = "docs/experiments.md", "docs/experiments.json", "docs/figures"
GROUPS = {
    "Наука (OpenAlex)": ["pub_log_total", "pub_growth_p", "pub_recent_share", "pub_age"],
    "Динамика новостей": ["news_log_recent", "news_growth_p", "news_saturated", "news_burst"],
    "Структура источников и деньги": ["funding_share", "pr_share", "mainstream_share", "domain_diversity"],
    "Код (GitHub)": ["gh_log_recent", "gh_log_total"],
    "Зрелость (Википедия, рынок)": ["wiki_age", "maturity_share"],
}
N_CONFIGS = {"n": 0}


# ---------------------------------------------------------------- базовые функции
def load() -> pd.DataFrame:
    df = pd.read_csv(IN)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        sys.exit(f"В {IN} нет признаков {missing}. Запустите: python -m scripts.build_training_set --recompute")
    df["errors"] = df["errors"].fillna("")
    return df


def fit(X, y, names, C=detector.GATE_C):
    return {"features": names, "gate": detector._fit_block(X, y, names, C=C, sign_constraints=True)}


def score(model, X, names):
    return np.array([detector.gate(dict(zip(names, x)), model)[0] for x in X])


def loto(df, names, C=detector.GATE_C, extra=None):
    """Вероятности вне выборки: на каждой теме — модель, обученная на остальных пяти."""
    X, y, areas = df[names].to_numpy(float), df.label.to_numpy(int), df.area.to_numpy()
    p = np.zeros(len(df))
    p_extra = np.zeros(len(extra)) if extra is not None else None
    for a in sorted(set(areas)):
        te = areas == a
        m = fit(X[~te], y[~te], names, C)
        p[te] = score(m, X[te], names)
        if extra is not None:
            ex = extra.area.to_numpy() == a
            if ex.any():
                p_extra[ex] = score(m, extra.loc[ex, names].to_numpy(float), names)
    N_CONFIGS["n"] += 1
    return (p, p_extra) if extra is not None else p


def metrics(y, p, thr=None):
    thr = detector.GATE_THRESHOLD if thr is None else thr
    pred = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p)}


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                      *["| " + " | ".join(str(c) for c in r) + " |" for r in rows]])


pc = lambda v: f"{v:.1%}"  # noqa: E731


# ---------------------------------------------------------------- эксперименты
def exp_main(df, p):
    return metrics(df.label.to_numpy(int), p)


def exp_random_cv(df, names=FEATURES):
    """Мягкая проверка: случайные фолды. Завышена, потому что близкие технологии из одной темы попадают
    и в обучение, и в проверку. Показывать только рядом со строгой оценкой и с этой оговоркой."""
    X, y = df[names].to_numpy(float), df.label.to_numpy(int)
    res = []
    for seed in range(3):
        p = np.zeros(len(df))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            p[te] = score(fit(X[tr], y[tr], names), X[te], names)
        res.append(metrics(y, p))
    return {k: float(np.mean([r[k] for r in res])) for k in res[0]}


def exp_ablation(df):
    y = df.label.to_numpy(int)
    full = metrics(y, loto(df, FEATURES))
    rows = []
    for g, cols in GROUPS.items():
        rest = [f for f in FEATURES if f not in cols]
        drop = metrics(y, loto(df, rest))
        only = metrics(y, loto(df, cols))
        rows.append({"group": g, "n_features": len(cols), "drop_f1": drop["f1"], "drop_auc": drop["roc_auc"],
                     "only_f1": only["f1"], "only_auc": only["roc_auc"], "delta_f1": full["f1"] - drop["f1"]})
    return full, rows


def exp_c_sweep(df):
    y = df.label.to_numpy(int)
    rows = []
    for C in (0.05, 0.1, 0.3, 1.0, 3.0):
        m = metrics(y, loto(df, FEATURES, C=C))
        rows.append({"C": C, **m})
    return rows


def exp_calibration(y, p):
    edges = [0, 0.2, 0.4, 0.6, 0.75, 0.9, 1.0001]
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append({"bin": f"{lo:.2f}–{min(hi, 1):.2f}", "n": int(m.sum()), "mean_p": float(p[m].mean()),
                         "frac_pos": float(y[m].mean())})
            ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    high = p >= 0.75
    return {"bins": rows, "ece": float(ece), "brier": float(brier_score_loss(y, p)),
            "n_high": int(high.sum()), "precision_high": float(y[high].mean()) if high.any() else None}


def exp_thresholds(y, p):
    return [{"thr": t, **{k: v for k, v in metrics(y, p, t).items() if k in ("accuracy", "precision", "recall", "f1")}}
            for t in (0.4, 0.5, detector.GATE_THRESHOLD, 0.64, 0.7, 0.75, 0.8, 0.9)]


def exp_negative_types(df, p):
    rows = []
    for kind, g in df.groupby("kind"):
        idx = g.index.to_numpy()
        if kind == "слабый сигнал":
            rows.append({"kind": kind, "n": len(g), "correct": float((p[idx] >= detector.GATE_THRESHOLD).mean()), "what": "recall (найден как сигнал)"})
        else:
            rows.append({"kind": kind, "n": len(g), "correct": float((p[idx] < detector.GATE_THRESHOLD).mean()), "what": "specificity (отсеян)"})
    return rows


def exp_pool_top15(df, p):
    """Ранжирование при идеальном извлечении кандидатов: пул темы = её слабые сигналы + негативы,
    берём 15 лучших по вероятности. Верхняя оценка качества шага отбора, а не сквозной результат."""
    prior = detector.prior_model()
    p_prior = np.array([detector.gate(r, prior)[0] for r in df.to_dict("records")])
    naive = df["news_growth_p"].to_numpy(float)
    rows = []
    for a in sorted(df.area.unique()):
        idx = np.where(df.area.to_numpy() == a)[0]
        y = df.label.to_numpy(int)[idx]
        k = min(15, len(idx))

        def at15(scores):
            top = np.argsort(-scores[idx])[:k]
            return int(y[top].sum())
        hits = {"model": at15(p), "prior": at15(p_prior), "naive": at15(naive)}
        rows.append({"area": a, "pool": len(idx), "positives": int(y.sum()), "k": k,
                     "hits_model": hits["model"], "hits_prior": hits["prior"], "hits_naive": hits["naive"],
                     "recall_model": hits["model"] / y.sum(), "precision_model": hits["model"] / k,
                     "random_expect": k * y.mean(), "ap": float(average_precision_score(y, p[idx]))})
    return rows


def exp_stage_trend(df):
    pos = df[df.label == 1].reset_index(drop=True)
    model = detector.train(df.to_dict("records"))["metrics"]
    rule = {"features": FEATURES, "stage": None, "trend": None}
    rs, rt = [], []
    for r in pos.to_dict("records"):
        s, t = detector.stage_trend(r, rule)
        rs.append(min(s, 4)); rt.append(3 if t == 3 else 2)
    rs, rt = np.array(rs), np.array(rt)
    ts, tt = pos.stage.to_numpy(int), np.where(pos.trend.to_numpy(int) == 3, 3, 2)
    return {"model": {"stage_exact": model["stage_exact"], "stage_within1": model["stage_within1"],
                      "trend_exact": model["trend_exact"], "score_within1": model["score_within1"]},
            "rule": {"stage_exact": float((rs == ts).mean()), "stage_within1": float((abs(rs - ts) <= 1).mean()),
                     "trend_exact": float((rt == tt).mean()),
                     "score_within1": float((abs((rs + rt) - (ts + np.where(pos.trend.to_numpy(int) == 3, 3, 2))) <= 1).mean())},
            "cv_gate": model["gate"]}


def exp_robustness(df):
    out = []
    pos = df[df.label == 1]
    for path in sorted(glob.glob("data/training/features_alt*.csv")):
        alt = pd.read_csv(path)
        alt = alt[alt.name_ru.isin(pos.name_ru)].reset_index(drop=True)
        if alt.empty:
            continue
        p, p_alt = loto(df, FEATURES, extra=alt)
        base = pd.Series(p, index=df.name_ru).groupby(level=0).first()
        p0 = base.loc[alt.name_ru].to_numpy()
        d = p_alt - p0
        out.append({"variant": os.path.basename(path).replace("features_", "").replace(".csv", ""), "n": len(alt),
                    "mean_abs_dp": float(np.abs(d).mean()), "median_abs_dp": float(np.median(np.abs(d))),
                    "flips": int(((p0 >= detector.GATE_THRESHOLD) != (p_alt >= detector.GATE_THRESHOLD)).sum()), "mean_p_main": float(p0.mean()),
                    "mean_p_alt": float(p_alt.mean()),
                    "worst": [{"query": f"{r.name_en} / {r.query_ru}", "p_main": round(float(a), 2), "p_alt": round(float(b), 2)}
                              for r, a, b in sorted(zip(alt.itertuples(), p0, p_alt), key=lambda t: -abs(t[2] - t[1]))[:3]]})
    return out


# ---------------------------------------------------------------- рисунки
def figures(res, df, p):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен — рисунки пропущены (pip install matplotlib)")
        return
    os.makedirs(FIG, exist_ok=True)
    # Палитра брендбука Aurum Foresight System: пергамент, чернила, латунь. Шрифт Montserrat, если установлен.
    from matplotlib import font_manager
    parchment, ink, accent, grey = "#EFEBDF", "#121B27", "#A8811F", "#8C8E95"
    family = "Montserrat" if any(f.name == "Montserrat" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.size": 12, "font.family": family, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": parchment, "axes.facecolor": parchment,
                         "savefig.facecolor": parchment, "text.color": ink, "axes.labelcolor": ink,
                         "axes.edgecolor": ink, "xtick.color": ink, "ytick.color": ink})

    rows = res["ablation"]["rows"]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [r["group"] for r in rows]
    ax.barh(labels, [r["delta_f1"] * 100 for r in rows], color=accent)
    ax.axvline(0, color=ink, lw=0.8)
    ax.set_xlabel("Потеря F1 при удалении группы признаков, п. п.")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(f"{FIG}/ablation.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    b = res["calibration"]["bins"]
    ax.plot([0, 1], [0, 1], color=grey, ls="--", label="идеальная калибровка")
    ax.plot([x["mean_p"] for x in b], [x["frac_pos"] for x in b], marker="o", color=accent, label="детектор")
    ax.axvline(0.75, color=ink, lw=0.6, ls=":")
    ax.set_xlabel("Уверенность модели"); ax.set_ylabel("Доля настоящих слабых сигналов"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/calibration.png", dpi=200); plt.close(fig)

    rows = res["pool_top15"]
    fig, ax = plt.subplots(figsize=(9, 4))
    x, w = np.arange(len(rows)), 0.2
    for i, (key, lab, col) in enumerate([("hits_model", "детектор", accent), ("hits_prior", "ручные веса", ink),
                                         ("hits_naive", "только рост новостей", grey)]):
        ax.bar(x + (i - 1) * w, [r[key] for r in rows], w, label=lab, color=col)
    ax.plot(x + 1.5 * w, [r["random_expect"] for r in rows], "_", color=ink, ms=18, label="случайный выбор")
    ax.set_xticks(x); ax.set_xticklabels([r["area"] for r in rows], rotation=15)
    ax.set_ylabel("Слабых сигналов в топ-15 пула"); ax.legend(frameon=False, ncol=2, fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/pool_top15.png", dpi=200); plt.close(fig)


def redraw_figures():
    """Перерисовать рисунки из уже сохранённого docs/experiments.json, без пересчёта метрик."""
    with open(OUT_JSON, encoding="utf-8") as fh:
        figures(json.load(fh), None, None)


# ---------------------------------------------------------------- запуск
def main():
    df = load()
    y = df.label.to_numpy(int)
    print(f"{len(df)} примеров: {int(y.sum())} слабых, {int((1 - y).sum())} негативов; ошибок источников в строках: "
          f"{int((df.errors != '').sum())}")
    p = loto(df, FEATURES)
    res = {"n": len(df), "n_weak": int(y.sum()), "n_neg": int((1 - y).sum()), "main": exp_main(df, p),
           "random_cv": exp_random_cv(df)}
    res["ablation"] = dict(zip(("full", "rows"), exp_ablation(df)))
    res["c_sweep"] = exp_c_sweep(df)
    res["calibration"] = exp_calibration(y, p)
    res["thresholds"] = exp_thresholds(y, p)
    res["negative_types"] = exp_negative_types(df, p)
    res["pool_top15"] = exp_pool_top15(df, p)
    res["stage_trend"] = exp_stage_trend(df)
    res["robustness"] = exp_robustness(df)
    res["n_configs"] = N_CONFIGS["n"]
    res["bad_rows"] = df.loc[df.errors != "", ["name_en", "errors"]].head(15).to_dict("records")
    json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    figures(res, df, p)

    m, rc = res["main"], res["random_cv"]
    share = float(y.mean())
    triv = {"accuracy": share, "precision": share, "recall": 1.0, "f1": 2 * share / (1 + share)}
    L = ["# Эксперименты", "",
         f"Выборка: {res['n']} строк. В ней {res['n_weak']} слабых сигналов и {res['n_neg']} негатива: "
         "зрелые технологии, хайп и общие понятия. В таблице указаны способы проверки. "
         f"Сравнено {res['n_configs']} конфигураций модели.", "",
         "## 1. Главная оценка детектора", "",
         md_table(["Схема проверки", "Accuracy", "Precision", "Recall", "F1", "ROC AUC (справка)"],
                  [["Валидация по ТЗ: 5 фолдов × 3 повтора, стратифицированная", pc(rc["accuracy"]), pc(rc["precision"]), pc(rc["recall"]), pc(rc["f1"]), f"{rc['roc_auc']:.3f}"],
                   ["Жёсткая проверка: модель не видела тему (leave-one-theme-out)", pc(m["accuracy"]), pc(m["precision"]), pc(m["recall"]), pc(m["f1"]), f"{m['roc_auc']:.3f}"],
                   ["Ориентир: всё считать слабым сигналом", pc(triv["accuracy"]), pc(triv["precision"]), pc(triv["recall"]), pc(triv["f1"]), "0.500"],
                   ["Ориентир: всё отклонить", pc(1 - triv["accuracy"]), "нет оценки", "0.0%", "0.0%", "0.500"]]),
         "", f"Итоговая модель: C={detector.GATE_C:g}, порог {detector.GATE_THRESHOLD:.2f}. "
             "Знаки восьми весов заданы при обучении. Эти знаки нельзя считать независимым выводом из данных.", "",
         "Валидация по ТЗ использует пять фолдов и три повтора с фиксированными random_state 0, 1 и 2. "
             "Похожие технологии могут попасть и в обучение, и в проверку. Поэтому рядом дана проверка на целиком новой теме. "
             f"Доля слабых сигналов в выборке {pc(triv['accuracy'])}. Если считать слабым сигналом всё подряд, F1 будет {pc(triv['f1'])}.", "",
         "## 2. Абляция: какие группы признаков отделяют сигнал от шума", "",
         md_table(["Группа", "Признаков", "F1 без группы", "Потеря F1", "F1 только на группе", "AUC только на группе"],
                  [[r["group"], r["n_features"], pc(r["drop_f1"]), f"{r['delta_f1'] * 100:+.1f} п. п.", pc(r["only_f1"]), f"{r['only_auc']:.3f}"]
                   for r in res["ablation"]["rows"]]), "",
         "## 3. Калибровка уверенности", "",
         md_table(["Интервал уверенности", "Примеров", "Средняя уверенность", "Доля настоящих сигналов"],
                  [[b["bin"], b["n"], pc(b["mean_p"]), pc(b["frac_pos"])] for b in res["calibration"]["bins"]]), "",
         f"ECE = {res['calibration']['ece']:.3f}, Brier = {res['calibration']['brier']:.3f}. "
         f"Среди {res['calibration']['n_high']} примеров с уверенностью ≥75% настоящими сигналами оказались "
         f"{pc(res['calibration']['precision_high']) if res['calibration']['precision_high'] is not None else 'нет оценки'}.", "",
         "## 4. Порог отнесения к слабым сигналам", "",
         md_table(["Порог", "Accuracy", "Precision", "Recall", "F1"],
                  [[t["thr"], pc(t["accuracy"]), pc(t["precision"]), pc(t["recall"]), pc(t["f1"])] for t in res["thresholds"]]), "",
         "## 5. Ошибки по типам примеров", "",
         md_table(["Тип", "Примеров", "Доля верных", "Что измеряет"],
                  [[r["kind"], r["n"], pc(r["correct"]), r["what"]] for r in res["negative_types"]]), "",
         "## 6. Ранжирование внутри готового пула", "",
         "Пул каждой темы состоит из её слабых сигналов и негативов. В таблице показаны 15 строк с наибольшей вероятностью. "
         "Поиск кандидатов здесь не проверялся.", "",
         md_table(["Тема", "Пул", "Сигналов", "Детектор в топ-15", "Ручные веса", "Только рост новостей", "Случайно", "Recall@15", "AP"],
                  [[r["area"], r["pool"], r["positives"], r["hits_model"], r["hits_prior"], r["hits_naive"],
                    f"{r['random_expect']:.1f}", pc(r["recall_model"]), f"{r['ap']:.2f}"] for r in res["pool_top15"]]), "",
         "## 7. Стадия и тренд против разметки датасета", "",
         md_table(["Метод", "Стадия точно", "Стадия ±1", "Тренд", "Балл ±1"],
                  [[k, pc(v["stage_exact"]), pc(v["stage_within1"]), pc(v["trend_exact"]), pc(v["score_within1"])]
                   for k, v in (("обученная модель", res["stage_trend"]["model"]), ("правило по лестнице зрелости", res["stage_trend"]["rule"]))]), "",
         "## 8. Подбор регуляризации", "",
         md_table(["C", "F1", "ROC AUC", "Brier"], [[r["C"], pc(r["f1"]), f"{r['roc_auc']:.3f}", f"{r['brier']:.3f}"] for r in res["c_sweep"]]), "",
         "Параметр C выбран на этих данных. При проверке на других данных результат может измениться.", ""]
    if res["robustness"]:
        L += ["## 9. Устойчивость к формулировке запроса", "",
              "Те же технологии проверены с другими поисковыми фразами. Δ показывает изменение вероятности.", "",
              md_table(["Вариант", "Технологий", "Среднее модуля Δ", "Медиана модуля Δ", "Сменили решение", "Ср. уверенность: основная → вариант"],
                       [[r["variant"], r["n"], f"{r['mean_abs_dp']:.2f}", f"{r['median_abs_dp']:.2f}", r["flips"],
                         f"{r['mean_p_main']:.2f} → {r['mean_p_alt']:.2f}"] for r in res["robustness"]]), ""]
    L += ["## Качество данных", "", f"Ошибки источников: {int((df.errors != '').sum())} из {len(df)} строк. "
          "Требование: не более 15% строк для каждого источника.", ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    redraw_figures() if "--figures-only" in sys.argv else main()
