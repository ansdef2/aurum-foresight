"""Обучение детектора и отчёт о метриках (кросс-валидация по темам).

    python -m scripts.train
"""
import json
import os

import pandas as pd

from app import config, detector
from app.features import FEATURES, LABELS
from scripts.experiments import exp_random_cv, md_table, pc

IN = "data/training/features.csv"
REPORT = "docs/metrics_detector.md"


def main():
    df = pd.read_csv(IN)
    rows = df.to_dict("records")
    model = detector.train(rows)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    json.dump(model, open(config.MODEL_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    m = model["metrics"]
    g = m["gate"]
    rc = exp_random_cv(df)
    lines = [
        "# Метрики детектора", "",
        "## Валидация по ТЗ (стандартная: 5 фолдов × 3 повтора, стратифицированная, random_state 0–2)", "",
        md_table(["Accuracy", "Precision", "Recall", "F1"], [[pc(rc["accuracy"]), pc(rc["precision"]), pc(rc["recall"]), pc(rc["f1"])]]), "",
        "Похожие технологии есть в нескольких темах. Поэтому проверка на случайных фолдах может дать завышенную оценку. "
        "Ниже дана проверка на новой теме.", "",
        "## Проверка на новой теме (leave-one-theme-out)", "",
        f"Строк в выборке: {m['n']}. Слабых сигналов: {m['n_weak']}. Негативов: {m['n_not_weak']}. "
        "При проверке leave-one-theme-out модель обучалась на пяти темах и проверялась на шестой.", "",
        "| Метрика | Детектор | Априорные веса (бейзлайн) |", "|---|---|---|",
        f"| Accuracy | {g['accuracy']:.1%} | {m['prior_baseline']['accuracy']:.1%} |",
        f"| Precision (слабый сигнал) | {g['precision']:.1%} | нет оценки |",
        f"| Recall (слабый сигнал) | {g['recall']:.1%} | нет оценки |",
        f"| F1 | {g['f1']:.1%} | {m['prior_baseline']['f1']:.1%} |",
        f"| ROC AUC | {g['roc_auc']:.3f} | нет оценки |" if g["roc_auc"] is not None else "", "",
        f"Матрица ошибок [[TN, FP], [FN, TP]]: {g['confusion']}", "",
        "## Стадия и тренд против разметки датасета", "",
        f"- Стадия: точное совпадение {m['stage_exact']:.1%}. Ошибка не больше 1: {m['stage_within1']:.1%}.",
        f"- Тренд (быстрый рост / рост): {m['trend_exact']:.1%}",
        f"- Балл «стадия + тренд»: точное совпадение {m['score_exact']:.1%}. Ошибка не больше 1: {m['score_within1']:.1%}.", "",
        "## По темам (тема не участвовала в обучении)", "", "| Тема | n | Accuracy | Recall слабых |", "|---|---|---|---|",
        *[f"| {a} | {v['n']} | {v['accuracy']:.1%} | {v['recall_weak']:.1%} |" for a, v in m["per_area"].items()], "",
        "## Веса признаков (итоговая модель)", "", "| Признак | Вес |", "|---|---|",
        *[f"| {LABELS[f]} | {w:+.3f} |" for f, w in sorted(zip(FEATURES, model["gate"]["coef"]), key=lambda x: -abs(x[1]))],
        "", "Знаки восьми весов заданы при обучении. Эти знаки нельзя считать независимым выводом из данных. "
        "Вес доли массовых СМИ близок к нулю и почти не влияет на решение.",
    ]
    open(REPORT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
