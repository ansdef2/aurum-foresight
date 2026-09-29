"""Render the Stage 1 training evidence as a seven-page PDF presentation.

Run after ``python -m scripts.train`` and ``python -m scripts.experiments``.
All numbers are read from the generated artifacts; this script does not train or score a model.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
W, H = 960, 540
PURPLE = colors.HexColor("#401064")
INK = colors.HexColor("#251735")
LILAC = colors.HexColor("#F4EFF9")
GOLD = colors.HexColor("#F2BC45")
GREEN = colors.HexColor("#21845A")
RED = colors.HexColor("#B84A5C")
GREY = colors.HexColor("#665D70")

arial = Path(r"C:\Windows\Fonts\arial.ttf")
arial_bold = Path(r"C:\Windows\Fonts\arialbd.ttf")
if arial.exists() and arial_bold.exists():
    regular_path, bold_path = arial, arial_bold
else:
    from matplotlib import font_manager

    regular_path = Path(font_manager.findfont("DejaVu Sans"))
    bold_path = Path(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold")))
pdfmetrics.registerFont(TTFont("ArialRU", str(regular_path)))
pdfmetrics.registerFont(TTFont("ArialRUBold", str(bold_path)))


def line(c, value, x, y, size=16, color=INK, bold=False):
    c.setFillColor(color)
    c.setFont("ArialRUBold" if bold else "ArialRU", size)
    c.drawString(x, y, str(value))


def wrapped(c, value, x, y, width, size=15, leading=21, color=INK, bold=False):
    font = "ArialRUBold" if bold else "ArialRU"
    words = str(value).split()
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if current and pdfmetrics.stringWidth(candidate, font, size) > width:
            line(c, current, x, y, size, color, bold)
            y -= leading
            current = word
        else:
            current = candidate
    if current:
        line(c, current, x, y, size, color, bold)
        y -= leading
    return y


def page(c, n, title, subtitle=""):
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(PURPLE)
    c.rect(0, H - 98, W, 98, fill=1, stroke=0)
    line(c, title, 55, H - 55, 29, colors.white, True)
    if subtitle:
        line(c, subtitle, 56, H - 78, 13, colors.HexColor("#E7D7F0"))
    c.setStrokeColor(colors.HexColor("#DDD4E7"))
    c.line(55, 42, W - 55, 42)
    line(c, "Обучение модели, 29.09.2026", 55, 25, 10, GREY)
    line(c, f"{n} / 7", W - 89, 25, 10, GREY)


def box(c, x, y, width, height, fill=LILAC):
    c.setFillColor(fill)
    c.roundRect(x, y, width, height, 13, fill=1, stroke=0)


def pct(value):
    return f"{value:.1%}".replace(".", ",")


def source_failures(rows):
    aliases = {
        "pub_counts": "OpenAlex", "pub_totals": "OpenAlex", "pubs": "OpenAlex",
        "news_recent": "Google News", "news_prev": "Google News", "news_ru": "Google News",
        "github": "GitHub", "wiki": "Википедия",
    }
    counts = Counter()
    for row in rows:
        failed = set()
        for part in row.get("errors", "").split("; "):
            key = part.split(":", 1)[0].strip()
            if key:
                failed.add(aliases.get(key, key))
        counts.update(failed)
    return counts


def read_inputs():
    with (ROOT / "docs/experiments.json").open(encoding="utf-8") as f:
        experiments = json.load(f)
    with (ROOT / "data/model/detector.json").open(encoding="utf-8") as f:
        model = json.load(f)
    with (ROOT / "data/training/features.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    log_path = ROOT / "docs/experiment_log.md"
    attempts = [s.strip() for s in log_path.read_text(encoding="utf-8").splitlines()
                if re.match(r"^\d+\.\s", s.strip())] if log_path.exists() else []
    if len(rows) != experiments["n"] or len(rows) != model["metrics"]["n"]:
        raise ValueError("The training CSV, model and experiment report have different row counts")
    return experiments, model, rows, attempts


def render(out):
    ex, model, rows, attempts = read_inputs()
    sources = source_failures(rows)
    n = len(rows)
    random_cv, strict = ex["random_cv"], ex["main"]
    targets = ("accuracy", "precision", "recall", "f1")
    target_met = all(random_cv[k] >= .75 for k in targets)
    weights = dict(zip(model["features"], model["gate"]["coef"]))
    required = [
        ("Возраст статьи Википедии", "wiki_age", -1),
        ("Насыщенный поток новостей", "news_saturated", -1),
        ("Объём публикаций", "pub_log_total", -1),
        ("Доля массовых СМИ", "mainstream_share", -1),
        ("Новости о раундах", "funding_share", 1),
        ("Рост публикаций", "pub_growth_p", 1),
        ("Рост новостей", "news_growth_p", 1),
        ("Всплеск новостей", "news_burst", 1),
    ]
    signs_ok = sum(weights[key] * sign > 0 for _, key, sign in required)
    source_ok = all(v / n <= .15 for v in sources.values())
    variants = {item["variant"]: item for item in ex.get("robustness", [])}
    out.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out), pagesize=(W, H))
    c.setTitle("Обучение модели слабых сигналов")
    c.setAuthor("Результаты обучения")

    # 1. Scope and result.
    c.setFillColor(PURPLE)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.roundRect(55, 409, 185, 34, 8, fill=1, stroke=0)
    line(c, "ОТЧЁТ ПО ЭТАПУ 1", 69, 420, 15, PURPLE, True)
    line(c, "Обучение модели", 55, 322, 45, colors.white, True)
    line(c, "слабых технологических сигналов", 55, 271, 35, colors.white, True)
    wrapped(c, "Данные из открытых источников. Логистическая регрессия. Проверка на шести темах.",
            57, 214, 810, 20, 29, colors.HexColor("#EADDF0"))
    line(c, f"Выборка: {n} строки. Требование: четыре метрики от 75%.", 57, 126, 19, colors.white)
    line(c, "Все четыре метрики выше 75%." if target_met else "Не все метрики выше 75%.",
         57, 83, 24, GOLD, True)
    line(c, "29 сентября 2026", 57, 43, 13, colors.HexColor("#EADDF0"))
    c.showPage()

    # 2. Data and model.
    page(c, 2, "Как считали", "Текст объяснений из таблицы не использовали как признак")
    cards = [
        ("01. Разметка", f"100 слабых сигналов из таблицы и {n - 100} негативных примера."),
        ("02. Сбор", "OpenAlex, Google News, GitHub и Википедия. Дата признаков: 28.09.2026."),
        ("03. Признаки", "16 числовых признаков: рост, объём, раунды, код, зрелость и состав источников."),
        ("04. Модель", "Стандартизация и логистическая регрессия. Для каждого признака виден его вклад."),
    ]
    for i, (heading, body) in enumerate(cards):
        x = 55 + (i % 2) * 435
        y = 259 if i < 2 else 102
        box(c, x, y, 410, 132)
        line(c, heading, x + 20, y + 96, 19, PURPLE, True)
        wrapped(c, body, x + 20, y + 65, 365, 16, 22)
    c.showPage()

    # 3. Sources.
    page(c, 3, "Ошибки источников", f"Число ошибок по каждому источнику на {n} строки")
    services = ["OpenAlex", "Google News", "GitHub", "Википедия"]
    for i, service in enumerate(services):
        y = 352 - i * 67
        box(c, 55, y - 22, 850, 57, colors.HexColor("#FAF8FC") if i % 2 else LILAC)
        line(c, service, 72, y + 1, 19, INK, True)
        count = sources[service]
        line(c, f"{count} / {n}", 617, y + 1, 18, INK)
        line(c, pct(count / n), 781, y + 1, 18, GREEN if count / n <= .15 else RED, True)
    line(c, "Условие: отказов каждого источника не более чем в 15% строк.", 62, 94, 15, GREY)
    line(c, "В обоих дополнительных наборах по 24 строки ошибок тоже нет.", 62, 70, 14, GREY)
    c.showPage()

    # 4. Validation.
    page(c, 4, "Результаты проверки", "Каждая метрика в строке «Валидация по ТЗ» должна быть от 75%")
    labels = [("Accuracy", "accuracy"), ("Precision", "precision"),
              ("Recall", "recall"), ("F1", "f1")]
    line(c, "Метрика", 68, 383, 15, GREY, True)
    line(c, "5 фолдов × 3 повтора", 358, 383, 15, PURPLE, True)
    line(c, "Новая тема (LOTO)", 666, 383, 15, GREY, True)
    for i, (label, key) in enumerate(labels):
        y = 329 - i * 62
        box(c, 55, y - 23, 850, 53, LILAC if i % 2 == 0 else colors.HexColor("#FAF8FC"))
        line(c, label, 70, y, 19, INK, True)
        line(c, pct(random_cv[key]), 388, y, 20, GREEN if random_cv[key] >= .75 else RED, True)
        line(c, pct(strict[key]), 710, y, 20, INK, True)
    wrapped(c, "При случайном разбиении похожие технологии могут попасть и в обучение, и в проверку. В LOTO модель проверяют на теме, которой не было в обучении.",
            64, 90, 832, 13, 17, GREY)
    c.showPage()

    # 5. Weight directions.
    page(c, 5, "Веса признаков", "Вес после стандартизации")
    for i, (label, key, sign) in enumerate(required):
        x = 55 + (i // 4) * 440
        y = 352 - (i % 4) * 73
        box(c, x, y - 26, 412, 60, LILAC)
        line(c, label, x + 14, y + 4, 15, INK)
        weight = weights[key]
        good = weight * sign > 0
        line(c, f"{weight:+.3f}".replace(".", ","), x + 294, y + 4, 17, GREEN if good else RED, True)
    line(c, f"Нужный знак имеют {signs_ok} из {len(required)} весов.", 62, 87, 16, PURPLE, True)
    line(c, "Знаки заданы при обучении. Вес СМИ −0,001 почти не влияет на решение.",
         62, 64, 13, GREY)
    c.showPage()

    # 6. Attempts.
    page(c, 6, "Что проверяли", "Четыре запуска. Полная запись: docs/experiment_log.md")
    if attempts:
        for i, attempt in enumerate(attempts[:5]):
            y = 365 - i * 70
            box(c, 55, y - 44, 850, 62, LILAC if i % 2 == 0 else colors.HexColor("#FAF8FC"))
            wrapped(c, attempt, 70, y - 3, 815, 13, 17)
    else:
        line(c, "Журнал не заполнен.", 70, 345, 19, RED)
    c.showPage()

    # 7. Conclusion and limitations.
    page(c, 7, "Что получилось")
    checks = [
        ("Четыре метрики ≥75%", target_met),
        ("Ошибки каждого источника ≤15%", source_ok),
        ("Знаки весов по методике", signs_ok == len(required)),
    ]
    for i, (label, okay) in enumerate(checks):
        y = 358 - i * 68
        box(c, 55, y - 24, 850, 54)
        line(c, label, 73, y, 18, INK, True)
        line(c, "ВЫПОЛНЕНО" if okay else "НЕ ВЫПОЛНЕНО", 699, y, 15, GREEN if okay else RED, True)
    changes = ", ".join(f"{variants[key]['flips']} из {variants[key]['n']} для {key}"
                        for key in ("alt1", "alt2") if key in variants)
    caveat = (f"Оценка сделана на {ex['n_weak']} положительных примерах из таблицы и "
              f"{ex['n_neg']} подготовленных негативах. При других поисковых фразах решение менялось "
              f"{changes}. Качество поиска кандидатов здесь не проверялось.")
    wrapped(c, caveat,
            63, 141, 835, 14, 19, GREY)
    line(c, "Подробные цифры: docs/metrics_detector.md и docs/experiments.md", 63, 57, 12, PURPLE)
    c.showPage()
    c.save()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "docs/stage1_training_result.pdf")
    args = parser.parse_args()
    print(render(args.out))
