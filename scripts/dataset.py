"""Загрузка эталонного датасета заказчика и разбор стадии/тренда в коды 1–4 / 1–3.

Используется ТОЛЬКО скриптами обучения и оценки. Пакет app/ этот файл не импортирует
(см. тест утечки в README)."""
import re

import pandas as pd

DATASET = "data/eval/dataset.xlsx"


def _stage_code(part: str):
    p = part.strip().lower()
    if "концепц" in p or p.startswith("исследован"):
        return 1
    if "прототип" in p or "poc" in p:
        return 2
    if "пилот" in p:
        return 3
    if "ранн" in p:
        return 4
    return None


def stage_code(text: str) -> int:
    codes = [c for c in (_stage_code(x) for x in re.split(r"→", str(text))) if c]
    return max(codes) if codes else 2


def trend_code(text: str) -> int:
    head = re.split(r"[—:–]", str(text))[0].lower()
    if "быстро" in head:
        return 3
    if "стабильн" in head or "медленно" in head:
        return 1
    return 2


def load(path: str = DATASET) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df = df.rename(columns={"Технология (слабый сигнал)": "name_ru", "Область": "area", "Компании": "companies",
                            "Почему это слабый сигнал": "why", "Стадия развития": "stage_text",
                            "Тренд упоминаний": "trend_text", "Балл (стадия+тренд)": "score", "Источники": "sources"})
    df["stage"] = df["stage_text"].map(stage_code)
    df["trend"] = df["trend_text"].map(trend_code)
    return df
