"""Настройки сервиса. Всё берётся из переменных окружения (.env)."""
import os


def env(name: str, default=None):
    value = os.environ.get(name)
    return default if value in (None, "") else value


import datetime as _dt

DATABASE_URL = env("DATABASE_URL", "sqlite:///./weak_signals.db")
CONTACT_EMAIL = env("CONTACT_EMAIL", "team@example.com")  # OpenAlex polite pool
GITHUB_TOKEN = env("GITHUB_TOKEN")
MODEL_PATH = env("MODEL_PATH", "data/model/detector.json")

# live — реальные источники; mock — синтетика для разработки интерфейса без сети
SOURCES_MODE = env("SOURCES_MODE", "live")

# Фиксированное назначение моделей по задачам (ТЗ: модели раскрыты, без автоматического выбора).
# Формат: провайдер:модель. Проверьте идентификаторы моделей в документации провайдера.
TASK_MODELS = {
    "expand": env("MODEL_EXPAND", "gigachat:GigaChat-2-Max"),      # разбор запроса на подобласти
    "extract": env("MODEL_EXTRACT", "gigachat:GigaChat-2"),        # извлечение кандидатов из заголовков
    "canonical": env("MODEL_CANONICAL", "gigachat:GigaChat-2"),    # поисковые формулировки для обучающей выборки
    "report": env("MODEL_REPORT", "gigachat:GigaChat-2-Max"),      # описание, преимущество, кейс, резюме на русском
    "judge": env("MODEL_JUDGE", "gigachat:GigaChat-2-Pro"),        # сопоставление с датасетом в скрипте оценки
}

# Разрешённые ТЗ модели. None = идентификатор не зафиксирован, сверить перед сдачей.
ALLOWED_MODELS = {
    "gigachat": {"GigaChat-2", "GigaChat-2-Pro", "GigaChat-2-Max"},
    "yandex": {"yandexgpt-lite", "yandexgpt"},          # YandexGPT Lite 5 / Pro 5 / 5.1
    "openai": {"gpt-4.1", "gpt-5.6-luna"},
    "qwen": None,                                        # Qwen3.6 35B-A3B, Qwen3 235B
    "mock": None,                                        # только для тестов и разработки
}

# Параметры конвейера
MAX_SUBTOPICS = int(env("MAX_SUBTOPICS", 10))
MAX_CANDIDATES = int(env("MAX_CANDIDATES", 60))
TOP_N = int(env("TOP_N", 15))
CONFIDENT = float(env("CONFIDENT", 0.75))
TIME_BUDGET_SEC = int(env("TIME_BUDGET_SEC", 17 * 60))
CACHE_TTL_HOURS = int(env("CACHE_TTL_HOURS", 24))
WORKERS = int(env("WORKERS", 8))

# Вызовы моделей: повторы при 429/5xx/таймаутах и потолок одновременных запросов (0 — без потолка)
LLM_RETRIES = int(env("LLM_RETRIES", 4))
LLM_MAX_CONCURRENCY = int(env("LLM_MAX_CONCURRENCY", 0))
LLM_TIMEOUT_SEC = int(env("LLM_TIMEOUT_SEC", 180))

# Фиксирует «сегодня» (ГГГГ-ММ-ДД). Для обучения задайте одну дату на весь прогон: тогда запросы к
# новостям одинаковы, кэш работает после полуночи, а признаки воспроизводимы.
AS_OF_DATE = env("AS_OF_DATE")


def today() -> _dt.date:
    return _dt.date.fromisoformat(AS_OF_DATE) if AS_OF_DATE else _dt.date.today()
