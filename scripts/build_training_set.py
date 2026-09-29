"""Обучающая выборка: 100 строк датасета (слабые сигналы) + негативы (зрелые, хайп, общие понятия).

Признаки считаются тем же сбором из открытых источников, что и в боевом конвейере, по названию
технологии. Текст колонок датасета («почему это слабый сигнал» и т. п.) в признаки НЕ попадает.

Перед запуском задайте в .env одну дату на весь прогон, например AS_OF_DATE=2026-09-28.
Тогда запросы к новостям одинаковы и кэш работает даже после полуночи.

    python -m scripts.build_training_set                    # основная выборка, докачка после обрыва
    python -m scripts.build_training_set --recompute        # пересчёт признаков из кэша (после правок features.py)
    python -m scripts.build_training_set --canon data/training/canonical_alt1.json \
        --out data/training/features_alt1.csv --only-positives   # другая формулировка запросов
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from app import config, db, features, llm, sources
from app.features import FEATURES
from scripts.dataset import load

OUT = "data/training/features.csv"
CANON = "data/training/canonical.json"
FIELDS = ["name_ru", "query_ru", "name_en", "area", "label", "stage", "trend", "kind", *FEATURES, "errors"]

CANON_SYSTEM = ("Ты помогаешь искать технологию в открытых источниках. Дай короткую поисковую формулировку "
                "на английском (2–5 слов), как эту технологию называют в новостях и статьях, и короткое "
                "название на русском.")
CANON_USER = "Технология: «{name}». Верни JSON: {{\"name_en\": \"...\", \"name_ru\": \"...\"}}"


def canonical_names(df: pd.DataFrame, path: str, generate: bool) -> dict:
    cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    if not generate:
        return cache
    for name in df["name_ru"]:
        if name in cache:
            continue
        try:
            data = llm.call_json("canonical", CANON_SYSTEM, CANON_USER.format(name=name))
            cache[name] = {"name_en": data.get("name_en") or name, "name_ru": data.get("name_ru") or name}
        except llm.LLMError as exc:
            print("canonical failed:", name, exc, file=sys.stderr)
            continue
        json.dump(cache, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon", default=CANON)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--only-positives", action="store_true")
    ap.add_argument("--recompute", action="store_true", help="игнорировать готовый CSV и пересчитать (из кэша)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    db.init_db()
    print(f"«сегодня» для признаков: {config.today()} "
          f"({'зафиксировано AS_OF_DATE' if config.AS_OF_DATE else 'системная дата, лучше задать AS_OF_DATE'})")
    df = load()
    canon = canonical_names(df, args.canon, generate=(args.canon == CANON))
    rows = []
    for _, r in df.iterrows():
        c = canon.get(r["name_ru"])
        if c:
            rows.append({"name_ru": r["name_ru"], "query_ru": c["name_ru"], "name_en": c["name_en"], "area": r["area"],
                         "label": 1, "stage": int(r["stage"]), "trend": int(r["trend"]), "kind": "слабый сигнал"})
    if not args.only_positives:
        for _, r in pd.read_csv("data/negatives.csv").iterrows():
            rows.append({"name_ru": r["name_ru"], "query_ru": r["name_ru"], "name_en": r["name_en"], "area": r["area"],
                         "label": 0, "stage": 5, "trend": 2, "kind": r["kind"]})

    done, mode = set(), "a"
    if os.path.exists(args.out) and not args.recompute:
        header = list(pd.read_csv(args.out, nrows=0).columns)
        if header != FIELDS:
            sys.exit(f"Набор признаков изменился с прошлого прогона ({args.out}). "
                     f"Запустите с --recompute: пересчёт идёт из кэша и занимает минуты.")
        done = set(pd.read_csv(args.out)["name_ru"])
    elif args.recompute:
        mode = "w"
    todo = [r for r in rows if r["name_ru"] not in done]
    print(f"всего {len(rows)}, осталось {len(todo)}")
    write_header = mode == "w" or not os.path.exists(args.out)

    with open(args.out, mode, newline="", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        futs = {pool.submit(sources.collect_evidence, r["name_en"], "", None, include_pubs=False): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            ev = fut.result()
            f = features.compute(ev)
            w.writerow({**r, **{k: round(v, 5) for k, v in f.items()}, "errors": "; ".join(ev["errors"])[:300]})
            fh.flush()
            print(f"[{i}/{len(todo)}] {r['name_en']}: ошибок {len(ev['errors'])}")


if __name__ == "__main__":
    main()
