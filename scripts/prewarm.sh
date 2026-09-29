#!/usr/bin/env bash
# Прогрев кэша: ставит в очередь запросы, чтобы на показе жюри они отдавались мгновенно.
# Запросы выполняются по одному. Использование: ./scripts/prewarm.sh https://ДОМЕН
BASE="${1:-http://localhost:8000}"
QUERIES=("Финтех" "Кибербезопасность" "Искусственный интеллект" "Биотехнологии" "Энергетика" "Новые материалы")
for q in "${QUERIES[@]}"; do
  curl -s -X POST "$BASE/api/search" -H 'Content-Type: application/json' \
    -d "{\"query\": \"$q\"}"; echo "  <- $q"
done
