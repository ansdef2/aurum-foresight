# Aurum Foresight System — поиск слабых технологических сигналов

Пользователь вводит технологическое направление в свободной форме («Финтех», «перспективные решения
в кибербезопасности»). Сервис собирает открытые источники на русском и английском, выделяет кандидатов,
проверяет каждого собственным детектором и возвращает ТОП-15 слабых сигналов с объяснением, отчётом
и источниками. Зрелые технологии, хайп и шум показываются отдельным списком с причинами исключения.

## Домены

Основной: aurum-foresight.ru
Резервный: 185-12-34-56.sslip.io

## Быстрый запуск

```bash
cp .env.example .env          # впишите WS_DOMAIN, ключ разрешённой модели и GITHUB_TOKEN
docker compose up -d --build  # PostgreSQL + API + сайт + Caddy (HTTPS)
open https://$WS_DOMAIN       # наружу сервис выходит только через Caddy
```

Развёртывание на сервере (в том числе Timeweb Cloud), сертификат GigaChat и типовые сбои — [DEPLOY.md](DEPLOY.md).

Проверка доступности источников и моделей:

```bash
docker compose --profile tools run --rm tools python -m scripts.smoke_sources
```

Без ключей и сети (разработка интерфейса): `SOURCES_MODE=mock` и `MODEL_*=mock:mock` в `.env`.
Данные в этом режиме синтетические.

## Обучение и оценка

Перед началом задайте в `.env` одну дату на весь прогон: `AS_OF_DATE=2026-09-28`.

```bash
# 1. Признаки для 100 сигналов датасета и негативов (долго, прогресс сохраняется, можно перезапускать)
docker compose --profile tools run --rm tools python -m scripts.build_training_set
# 1b. Те же 24 технологии по другим формулировкам (проверка устойчивости к формулировке)
docker compose --profile tools run --rm tools python -m scripts.build_training_set --canon data/training/canonical_alt1.json --out data/training/features_alt1.csv --only-positives
docker compose --profile tools run --rm tools python -m scripts.build_training_set --canon data/training/canonical_alt2.json --out data/training/features_alt2.csv --only-positives
# 2. Обучение детектора + отчёт docs/metrics_detector.md (кросс-валидация по темам)
docker compose --profile tools run --rm tools python -m scripts.train
docker compose restart api    # API подхватит data/model/detector.json
# 3. Эксперименты: абляция, калибровка, пороги, ранжирование, устойчивость -> docs/experiments.md и docs/figures/
docker compose --profile tools run --rm tools python -m scripts.experiments
# 4. Сквозная оценка по 6 темам датасета в трёх формулировках запроса (нужен ключ модели)
docker compose --profile tools run --rm tools python -m scripts.evaluate --variants short tz en
```

После любой правки `app/features.py` пересчёт признаков идёт из кэша, без сети:
`python -m scripts.build_training_set --recompute` (с той же `AS_OF_DATE`).

## Временные метрики

| Процесс | Где | Время |
|---|---|---|
| Сборка обучающей выборки | офлайн, один раз | десятки минут, журнал `docs/build_training_set_run.log` |
| Обучение детектора | офлайн | секунды |
| Обработка одного запроса | онлайн | в пределах бюджета 17 мин (`TIME_BUDGET_SEC`) |

## Честность оценки

- Эталонный датасет заказчика в репозиторий не входит. Для переобучения положите его в `data/eval/dataset.xlsx`; читают его только скрипты обучения и оценки.
  Пакет `app/` его не импортирует (тест `test_no_dataset_leak_into_app`), а `.dockerignore`
  не пускает его в образ сервиса: работающий стенд физически не имеет доступа к ответам.
- Детектор проверяется по схеме leave-one-theme-out: метрика для темы получена моделью,
  которая эту тему не видела.
- Таблица сопоставления выдачи с датасетом публикуется целиком (`docs/eval_matches.csv`).

## Модели

Назначение моделей фиксировано и раскрыто в `.env` (`MODEL_*`), автоматического выбора нет.
Каждый вызов пишется в таблицу `llm_log` и доступен по `GET /api/jobs/{id}/llm`.
Языковая модель извлекает факты, переводит и пишет тексты только по найденным источникам.
Решение «слабый сигнал или нет» и уверенность считает детектор (`app/detector.py`).

## API

| Метод | Назначение |
|---|---|
| `POST /api/search {"query": "...", "force": false}` | запустить поиск (результат за 24 ч берётся из кэша) |
| `GET /api/jobs/{id}` | статус, этап, прогресс, результат |
| `GET /api/jobs/{id}/export.xlsx` | выдача в формате датасета заказчика |
| `GET /api/jobs/{id}/llm` | журнал вызовов моделей |
| `GET /api/model` | веса детектора, порог и метрики |
| `GET /api/health` | проверка стенда |

## Структура

```
app/            сервис: источники, признаки, детектор, конвейер, API, интерфейс
scripts/        обучающая выборка, обучение, сквозная оценка, проверка источников
data/eval/      сюда кладётся датасет заказчика (в репозитории отсутствует)
data/negatives.csv  негативные примеры: зрелые технологии, хайп, общие понятия
docs/           методология, архитектура, отчёты о метриках
```

Тесты: `pytest -q`.
