"""Кандидаты в слабые сигналы: разбор запроса → сбор документов → извлечение классов технологий → склейка."""
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config, llm, sources

GENERIC = {"artificial intelligence", "ai", "machine learning", "deep learning", "llm", "large language models",
           "generative ai", "blockchain", "robotics", "robots", "fintech", "cybersecurity", "edge computing",
           "cloud computing", "data", "big data", "automation", "digital transformation", "iot",
           "internet of things", "ai security", "ai infrastructure", "industrial ai", "neural networks",
           "computer vision", "nlp", "ai agents", "agents", "quantum computing", "semiconductors"}

EXPAND_SYSTEM = (
    "Ты технологический аналитик. Пользователь задал направление для поиска зарождающихся технологий. "
    "Разложи его на конкретные подобласти, в которых сейчас появляются новые классы решений. "
    "Для каждой подобласти дай короткий поисковый запрос на английском (2–5 слов) и название на русском.")
EXPAND_USER = ("Направление: «{query}».\nВерни JSON: {{\"theme_ru\": \"...\", \"subtopics\": "
               "[{{\"ru\": \"...\", \"en\": \"...\"}}]}}. Подобластей: {n}.")

EXTRACT_SYSTEM = (
    "Ты извлекаешь из заголовков новостей и научных работ названия конкретных классов технологий или решений. "
    "Класс технологии — это то, что аналитик внёс бы в радар: «LLM-файрволы на сетевом уровне», "
    "«маркетплейсы навыков для роботов», «конфиденциальный инференс в TEE». Не извлекай названия компаний, "
    "продуктов и слишком общие области вроде «искусственный интеллект» или «робототехника». "
    "Используй только то, что есть в заголовках, ничего не добавляй от себя.")
EXTRACT_USER = ("Направление: «{query}».\nЗаголовки:\n{lines}\n\nВерни JSON: {{\"items\": [{{\"name_ru\": \"...\", "
                "\"name_en\": \"поисковая фраза на английском, 2–5 слов\", \"doc_ids\": [номера заголовков]}}]}}")


def expand_query(query: str) -> dict:
    try:
        data = llm.call_json("expand", EXPAND_SYSTEM, EXPAND_USER.format(query=query, n=config.MAX_SUBTOPICS))
        subs = [s for s in data.get("subtopics", []) if s.get("en")][: config.MAX_SUBTOPICS]
        if subs:
            return {"theme_ru": data.get("theme_ru") or query, "subtopics": subs}
    except llm.LLMError:
        pass
    return {"theme_ru": query, "subtopics": [{"ru": query, "en": query}]}


def discover(subtopics: list[dict]) -> tuple[list, list]:
    """Документы по всем подобластям за последние ~18 месяцев. Возвращает (документы, ошибки)."""
    if config.SOURCES_MODE == "mock":
        from . import mock
        return mock.discover(subtopics), []
    tasks = []
    for s in subtopics:
        en, ru = s.get("en", ""), s.get("ru", "")
        tasks += [("news_en", lambda q=en: sources.gnews(q, "en", 540, 0)),
                  ("openalex", lambda q=en: sources.openalex_recent(q, 18, 25)),
                  ("arxiv", lambda q=en: sources.arxiv_recent(q, 25))]
        if ru:
            tasks.append(("news_ru", lambda q=ru: sources.gnews(q, "ru", 540, 0)))
    docs, errors = [], []
    with ThreadPoolExecutor(config.WORKERS) as pool:
        futures = [(name, pool.submit(fn)) for name, fn in tasks]
        for name, fut in futures:
            try:
                docs.extend(fut.result())
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}"[:200])
    unique = {}
    for d in docs:
        unique.setdefault(d.id, d)
    return list(unique.values()), errors


def extract(query: str, docs: list, batch: int = 40) -> list[dict]:
    """LLM извлекает кандидатов из заголовков партиями; каждый кандидат привязан к документам."""
    batches = [docs[i:i + batch] for i in range(0, len(docs), batch)]

    def run(chunk):
        lines = "\n".join(f"[{i}] {d.title} ({d.domain}, {d.date[:7]})" for i, d in enumerate(chunk))
        try:
            data = llm.call_json("extract", EXTRACT_SYSTEM, EXTRACT_USER.format(query=query, lines=lines),
                                 max_tokens=3000)
        except llm.LLMError:
            return []
        items = data.get("items", []) if isinstance(data, dict) else data
        out = []
        for it in items or []:
            ids = [chunk[i].id for i in it.get("doc_ids", []) if isinstance(i, int) and 0 <= i < len(chunk)]
            if it.get("name_en") and ids:
                out.append({"name_ru": (it.get("name_ru") or it["name_en"]).strip(),
                            "name_en": it["name_en"].strip(), "doc_ids": ids})
        return out

    items = []
    with ThreadPoolExecutor(min(4, config.WORKERS)) as pool:
        for res in pool.map(run, batches):
            items.extend(res)
    return items


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9а-яё\s-]", " ", s.lower())).strip()


def cluster(items: list[dict], docs_by_id: dict, threshold: float = 0.55) -> tuple[list[dict], list[dict]]:
    """Склейка синонимов: косинусная близость TF-IDF по символьным n-граммам английских формулировок
    плюс коэффициент Шимкевича — Симпсона по множествам документов. Возвращает (кандидаты, отсеянные)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    rejected = []
    kept = []
    for it in items:
        if _norm(it["name_en"]) in GENERIC or len(_norm(it["name_en"])) < 4:
            rejected.append({"name_ru": it["name_ru"], "name_en": it["name_en"], "category": "шум",
                             "reason": "слишком общее понятие, а не класс технологии", "confidence": None})
        else:
            kept.append(it)
    if not kept:
        return [], rejected
    names = [_norm(it["name_en"]) for it in kept]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(names)
    M = vec.transform(names)
    sim = (M @ M.T).toarray()
    sets = [set(it["doc_ids"]) for it in kept]
    parent = list(range(len(kept)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            overlap = len(sets[i] & sets[j]) / max(1, min(len(sets[i]), len(sets[j])))
            if 0.45 * sim[i, j] + 0.55 * overlap >= threshold or sim[i, j] >= 0.8:
                parent[find(i)] = find(j)

    groups: dict = {}
    for i in range(len(kept)):
        groups.setdefault(find(i), []).append(i)
    candidates = []
    for members in groups.values():
        doc_ids = sorted(set().union(*(sets[i] for i in members)))
        best = max(members, key=lambda i: len(sets[i]))
        domains = {docs_by_id[d].domain for d in doc_ids if d in docs_by_id}
        candidates.append({"name_ru": kept[best]["name_ru"], "name_en": kept[best]["name_en"],
                           "aliases": sorted({kept[i]["name_en"] for i in members} - {kept[best]["name_en"]}),
                           "doc_ids": doc_ids, "support": len(doc_ids), "domains": len(domains)})
    candidates.sort(key=lambda c: (c["domains"], c["support"]), reverse=True)
    return candidates, rejected


def support_score(c: dict) -> float:
    return float(np.log1p(c["support"]) + np.log1p(c["domains"]))
