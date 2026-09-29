"""Сквозная оценка так же, как будет проверять заказчик: открытый запрос по каждой из 6 тем датасета →
ТОП-15 → сопоставление со строками темы → метрики и таблица сопоставления для ручной проверки.

    python -m scripts.evaluate                 # все 6 тем, с проверкой совпадений моделью judge
    python -m scripts.evaluate --no-judge      # только текстовая близость (быстро, для итераций)
    python -m scripts.evaluate --themes Финтех Роботы
"""
import argparse
import json
import os
import time
import uuid

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app import db, llm, pipeline
from scripts.dataset import load

# Три формулировки запроса на тему. Формулировки фиксированы заранее и НЕ подбираются под результат:
# в отчёте идут среднее и разброс по формулировкам, а не лучшая.
QUERY_VARIANTS = {
    "short": {"Индустриальный ИИ": "Индустриальный ИИ", "Роботы": "Роботы", "Инфраструктура ИИ": "Инфраструктура ИИ",
              "Финтех": "Финтех", "Защита ИИ": "Защита ИИ", "Edge": "Edge AI"},
    "tz": {"Индустриальный ИИ": "технологии в области индустриального ИИ", "Роботы": "перспективные решения в робототехнике",
           "Инфраструктура ИИ": "слабые сигналы в области инфраструктуры ИИ", "Финтех": "перспективные решения в финтехе",
           "Защита ИИ": "слабые сигналы в области безопасности ИИ",
           "Edge": "технологии в области edge-вычислений и ИИ на устройствах"},
    "en": {"Индустриальный ИИ": "Industrial AI", "Роботы": "Robotics", "Инфраструктура ИИ": "AI infrastructure",
           "Финтех": "Fintech", "Защита ИИ": "AI security", "Edge": "Edge AI"},
}
QUERIES = QUERY_VARIANTS["short"]
OUT_DIR = "data/eval/results"
JUDGE_SYSTEM = ("Ты сопоставляешь две формулировки технологий. Совпадение — если это один и тот же класс "
                "технологий или решений, даже если названия разные или на разных языках. Если одна формулировка "
                "заметно шире или про другое применение — не совпадение.")
JUDGE_USER = ("A (эталон): «{a}». Пояснение эталона: {why}\nB (выдача системы): «{b}» ({b_en}). Описание B: {desc}\n"
              "Верни JSON: {{\"match\": true/false, \"reason\": \"кратко\"}}")


def run_theme(theme: str, variant: str = "short") -> dict:
    job_id = "eval-" + uuid.uuid4().hex[:10]
    q = QUERY_VARIANTS[variant][theme]
    db.create_job(job_id, q, q.lower())
    t0 = time.monotonic()
    result = pipeline.run(job_id, q)
    db.update_job(job_id, status="done", stage="Готово", progress=1.0, result=result, finished_at=db.now())
    result["eval_job_id"], result["eval_seconds"] = job_id, round(time.monotonic() - t0, 1)
    return result


def match(theme: str, rows: pd.DataFrame, result: dict, use_judge: bool, variant: str = "short", top_k: int = 3):
    signals = result["signals"]
    if not signals:
        return [], {}
    a_txt = rows["name_ru"].tolist()
    b_txt = [f"{s['name_ru']} {s['name_en']} {' '.join(s['aliases'])}" for s in signals]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(a_txt + b_txt)
    sim = (vec.transform(a_txt) @ vec.transform(b_txt).T).toarray()
    pairs = []
    for i, (_, r) in enumerate(rows.iterrows()):
        for j in np.argsort(-sim[i])[:top_k]:
            s = signals[j]
            verdict, reason = None, ""
            if use_judge:
                try:
                    out = llm.call_json("judge", JUDGE_SYSTEM, JUDGE_USER.format(
                        a=r["name_ru"], why=str(r["why"])[:300], b=s["name_ru"], b_en=s["name_en"],
                        desc=(s.get("description") or "")[:300]))
                    verdict, reason = bool(out.get("match")), out.get("reason", "")
                except llm.LLMError as exc:
                    reason = f"judge error: {exc}"
            else:
                verdict = sim[i, j] >= 0.35
            pairs.append({"theme": theme, "variant": variant, "dataset_no": int(r["№"]), "dataset_name": r["name_ru"],
                          "rank": s["rank"], "signal": s["name_ru"], "signal_en": s["name_en"],
                          "similarity": round(float(sim[i, j]), 3), "match": verdict, "reason": reason})
    # один к одному: жадно по близости среди подтверждённых пар
    used_rows, used_sig, final = set(), set(), []
    for p in sorted([p for p in pairs if p["match"]], key=lambda p: -p["similarity"]):
        if p["dataset_no"] in used_rows or p["rank"] in used_sig:
            continue
        used_rows.add(p["dataset_no"]); used_sig.add(p["rank"]); final.append(p)
    metrics = {"theme": theme, "variant": variant, "query": QUERY_VARIANTS[variant][theme], "dataset_rows": len(rows), "returned": len(signals),
               "hits": len(final), "recall": len(final) / len(rows), "precision_lower_bound": len(final) / len(signals),
               "seconds": result.get("eval_seconds")}
    return pairs, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--themes", nargs="*", default=list(QUERIES))
    ap.add_argument("--variants", nargs="*", default=["short"], choices=list(QUERY_VARIANTS))
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="не запускать поиск, взять сохранённые результаты")
    args = ap.parse_args()
    db.init_db()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()
    all_pairs, all_metrics = [], []
    for variant in args.variants:
        for theme in args.themes:
            path = os.path.join(OUT_DIR, f"{theme}.{variant}.json")
            if args.reuse and os.path.exists(path):
                result = json.load(open(path, encoding="utf-8"))
            else:
                print(f"== {theme} [{variant}]: запускаю поиск")
                result = run_theme(theme, variant)
                json.dump(result, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
            pairs, metrics = match(theme, df[df["area"] == theme], result, not args.no_judge, variant)
            all_pairs += pairs
            all_metrics.append(metrics)
            print(metrics)
    pd.DataFrame(all_pairs).to_csv("docs/eval_matches.csv", index=False)
    m = pd.DataFrame(all_metrics)
    lines = ["# Сквозная оценка по темам датасета", "",
             "Каждая тема подаётся как открытый запрос в нескольких формулировках. ТОП-15 сопоставляется со строками "
             "этой темы: кандидаты пар по текстовой близости, решение о совпадении — модель judge, затем ручная проверка "
             "по файлу `docs/eval_matches.csv`. Точность против датасета — нижняя граница: по словам заказчика, "
             "в каждой теме слабых сигналов больше 15, поэтому несовпавшая позиция не обязательно ошибочна. "
             "Формулировки зафиксированы заранее; в отчёт идут среднее и разброс, а не лучшая.", "",
             "| Тема | Формулировка | Запрос | Строк в датасете | Совпало в ТОП-15 | Полнота | Точность (нижняя граница) | Время, мин |",
             "|---|---|---|---|---|---|---|---|",
             *[f"| {r.theme} | {r.variant} | {r.query} | {r.dataset_rows} | {r.hits} | {r.recall:.0%} | "
               f"{r.precision_lower_bound:.0%} | {(r.seconds or 0) / 60:.1f} |" for r in m.itertuples()], ""]
    if m.variant.nunique() > 1:
        g = m.groupby("theme").recall.agg(["mean", "min", "max"])
        lines += ["## Устойчивость к формулировке запроса", "", "| Тема | Средняя полнота | Минимум | Максимум |", "|---|---|---|---|",
                  *[f"| {t} | {r['mean']:.0%} | {r['min']:.0%} | {r['max']:.0%} |" for t, r in g.iterrows()], ""]
    lines.append(f"Средняя полнота по всем запускам: {m.recall.mean():.0%}; максимум по формулировкам не используется как результат.")
    open("docs/metrics_end_to_end.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
