"""Быстрая проверка доступности всех источников и моделей. Запускать первым делом после настройки .env.

    python -m scripts.smoke_sources
"""
from app import config, db, llm, sources

db.init_db()
Q = "agent identity management"
checks = [
    ("OpenAlex: счётчики по годам", lambda: sources.openalex_counts(Q, ttl=0)),
    ("OpenAlex: объём корпуса", lambda: len(sources.openalex_totals(ttl=0))),
    ("OpenAlex: свежие работы", lambda: [d.title for d in sources.openalex_recent(Q, ttl=0)][:2]),
    ("arXiv", lambda: [d.title for d in sources.arxiv_recent(Q, 5, ttl=0)][:2]),
    ("Google News EN", lambda: [(d.title, d.domain, d.date) for d in sources.gnews(Q, "en", ttl=0)][:2]),
    ("Google News RU", lambda: [(d.title, d.domain, d.date) for d in sources.gnews("ИИ-агенты", "ru", ttl=0)][:2]),
    ("GitHub", lambda: {k: v for k, v in sources.github_stats(Q, ttl=0).items() if k != "top"}),
    ("Википедия", lambda: sources.wikipedia("Web application firewall")),
]
for name, fn in checks:
    try:
        print(f"OK   {name}: {str(fn())[:160]}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name}: {exc}")
if any(spec.startswith("gigachat:") for spec in config.TASK_MODELS.values()):
    # Идентификаторы в MODEL_* должны совпасть со списком, который отдаёт ключу сам GigaChat
    try:
        available = llm.gigachat_models()
        print(f"OK   GigaChat: модели, доступные ключу: {', '.join(available)}")
        for task, spec in config.TASK_MODELS.items():
            provider, model = spec.split(":", 1)
            if provider == "gigachat" and model not in available:
                print(f"FAIL MODEL_{task.upper()}={spec}: такого идентификатора нет в списке GigaChat")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL GigaChat: список моделей недоступен: {exc}")
for task in config.TASK_MODELS:
    try:
        print(f"OK   модель {task} ({config.TASK_MODELS[task]}): {llm.call(task, 'Ответь одним словом.', 'Скажи: готов')[:40]}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL модель {task} ({config.TASK_MODELS[task]}): {exc}")
