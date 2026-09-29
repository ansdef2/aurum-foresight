"""Отчёт по сигналу. Тексты (описание, преимущество, кейс, резюме зарубежных источников) пишет модель
строго по найденным источникам. Объяснение статуса и уверенности считается детектором, а не моделью."""
from . import llm
from .detector import STAGE_LABELS, TREND_LABELS

TRUST_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}
TRUST_LABELS = {"A": "высокий", "B": "средний", "C": "базовый", "D": "пониженный"}

REPORT_SYSTEM = (
    "Ты технологический аналитик банка. Пиши по-русски, коротко и конкретно. Используй только факты из "
    "переданных источников и ничего не добавляй от себя. Если факта нет в источниках — оставь поле пустым.")
REPORT_USER = (
    "Технология: «{name_ru}» ({name_en}).\nИсточники:\n{lines}\n\n"
    "Верни JSON: {{\"description\": \"что это за технология, 2–3 предложения\", "
    "\"advantage\": \"какую проблему решает и чем лучше существующих решений, 1–2 предложения\", "
    "\"case\": {{\"text\": \"конкретный кейс: кто, что сделал, когда\", \"source_id\": \"id источника\"}}, "
    "\"companies\": [\"компании и организации из источников\"], "
    "\"analyst_notes\": \"оценки из аналитических отчётов, если есть в источниках, иначе пусто\", "
    "\"summaries\": {{\"id источника\": \"резюме на русском в 1 предложение\"}}}}. "
    "Резюме дай для каждого источника не на русском языке.")


def pick_sources(doc_ids: list, docs_by_id: dict, evidence: dict, limit: int = 8) -> list:
    pool = [docs_by_id[i] for i in doc_ids if i in docs_by_id]
    for key in ("news_recent", "news_ru", "pubs"):
        pool += list(evidence.get(key) or [])[:5]
    pool += list((evidence.get("github") or {}).get("top") or [])[:1]
    seen, out = set(), []
    for d in sorted(pool, key=lambda d: (TRUST_ORDER.get(d.trust, 4), -(int(d.date[:4]) if d.date[:4].isdigit() else 0))):
        key = d.url or d.title
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out[:limit]


def build(candidate: dict, srcs: list) -> dict:
    lines = "\n".join(f"[{d.id}] {d.title} — {d.domain}, {d.date}, язык: {d.lang}. {d.snippet[:200]}" for d in srcs)
    model_spec = None
    try:
        from .llm import model_for
        model_spec = ":".join(model_for("report"))
        data = llm.call_json("report", REPORT_SYSTEM, REPORT_USER.format(
            name_ru=candidate["name_ru"], name_en=candidate["name_en"], lines=lines), max_tokens=2500)
    except llm.LLMError:
        data = {}
    ids = {d.id for d in srcs}
    case = data.get("case") or {}
    if case.get("source_id") not in ids:
        case = {"text": case.get("text", ""), "source_id": None,
                "note": "кейс не удалось привязать к источнику" if case.get("text") else ""}
    summaries = data.get("summaries") or {}
    out_sources = []
    for d in srcs:
        s = summaries.get(d.id, "") if d.lang != "ru" else ""
        out_sources.append({
            "id": d.id, "title": d.title, "url": d.url, "date": d.date or "не указана", "type": d.type,
            "lang": d.lang, "trust": d.trust, "trust_label": TRUST_LABELS.get(d.trust, "—"), "domain": d.domain,
            "summary_ru": s, "summary_note": f"генеративное резюме, модель {model_spec}" if s else ""})
    return {
        "description": data.get("description", ""), "advantage": data.get("advantage", ""), "case": case,
        "companies": [c for c in (data.get("companies") or []) if isinstance(c, str)][:6],
        "analyst_notes": data.get("analyst_notes", ""), "sources": out_sources, "report_model": model_spec}


def why_weak(stage: int, trend: int, expl: dict, p: float) -> str:
    pros = "; ".join(x["text"] for x in expl["pros"][:3]) or "нет сильных признаков"
    cons = "; ".join(x["text"] for x in expl["cons"][:2])
    text = (f"Стадия — {STAGE_LABELS[stage].lower()}, динамика — {TREND_LABELS[trend].lower()}. "
            f"За отнесение к слабым сигналам: {pros}.")
    if cons:
        text += f" Снижают уверенность: {cons}."
    text += f" Итоговая уверенность модели {p:.0%}."
    return text
