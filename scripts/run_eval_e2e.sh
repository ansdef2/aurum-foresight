#!/usr/bin/env bash
# Детектор уже обучен (data/model/detector.json). Остаётся только сквозная оценка по 6 темам (нужен ключ GigaChat).
# Запуск в фоне: nohup ./scripts/run_eval_e2e.sh > eval.log 2>&1 &
set -e
docker compose --profile tools run --rm tools python -m scripts.evaluate --variants short   # -> docs/metrics_end_to_end.md
