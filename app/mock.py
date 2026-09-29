"""Офлайн-режим (SOURCES_MODE=mock, модели mock:*) для разработки интерфейса и тестов без сети и ключей.
Данные синтетические и в отчётах о метриках не используются."""
import hashlib
import json
import re

from .sources import Document

WEAK = ["agent identity management", "mcp server security scanning", "kv cache offloading",
        "optical circuit switching", "robot skill marketplace", "teleoperation data services",
        "llm watermarking for text", "confidential inference attestation", "tactile vla models",
        "federated aml learning", "local currency stablecoins", "neuromorphic edge processors",
        "in-sensor computing", "machine unlearning services", "orbital edge inference",
        "ai compilers for inference", "autonomous procurement agents", "welding vision autonomy"]
MATURE = ["contactless payments", "cloud gpu instances", "industrial robot arms", "web application firewall",
          "smartphone npu", "robotic vacuum cleaners"]
DOMAINS = ["techcrunch.com", "siliconangle.com", "eetimes.com", "prnewswire.com", "reuters.com", "arxiv.org",
           "theaiinsider.tech", "cnews.ru", "geekwire.com", "sifted.eu"]


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def discover(subtopics):
    docs = []
    for tech in WEAK + MATURE:
        k = 3 + _h(tech) % 5
        for i in range(k):
            dom = DOMAINS[(_h(tech) + i) % len(DOMAINS)]
            verb = ["raises $12M for", "pilots", "introduces", "study on", "benchmark of"][i % 5]
            docs.append(Document(source="news", url=f"https://{dom}/{tech.replace(' ', '-')}/{i}",
                                 title=f"Startup {verb} {tech}", date=f"2026-0{1 + i % 8}-1{i % 9}",
                                 domain=dom, lang="en"))
    return docs


def evidence(name_en: str, name_ru: str = ""):
    mature = name_en in MATURE
    h = _h(name_en)
    counts = {y: (5000 + h % 3000 if mature else max(0, (y - 2021) * (2 + h % 5))) for y in range(2014, 2027)}
    totals = {y: 9_000_000 + (y - 2014) * 300_000 for y in range(2014, 2027)}
    n_recent = 100 if mature else 8 + h % 30
    n_prev = 90 if mature else 2 + h % 6
    news = [Document(source="news", url=f"https://{DOMAINS[(h + i) % len(DOMAINS)]}/{name_en}/{i}",
                     title=("Market leaders in " if mature else "Startup raises seed for ") + name_en,
                     date="2026-05-01", domain=DOMAINS[(h + i) % len(DOMAINS)], lang="en") for i in range(n_recent)]
    prev = news[:n_prev]
    return {"name_en": name_en, "name_ru": name_ru, "errors": [], "pub_counts": counts, "pub_totals": totals,
            "pubs": [Document(source="openalex", url=f"https://openalex.org/W{h % 10**8}", title=f"A study of {name_en}",
                              date="2025-11-02", lang="en")],
            "news_recent": news, "news_prev": prev, "news_ru": [],
            "github": {"total": 20000 if mature else h % 40, "recent": 300 if mature else h % 25, "top": []},
            "wiki": {"exists": mature, "created_year": 2009 if mature else None, "url": None, "title": None}}


def llm_response(system: str, user: str) -> str:
    if "Разложи его" in system:
        q = re.search(r"«(.+?)»", user).group(1)
        return json.dumps({"theme_ru": q, "subtopics": [{"ru": q, "en": q}]}, ensure_ascii=False)
    if "извлекаешь" in system:
        items = []
        for line in user.splitlines():
            m = re.match(r"\[(\d+)\] (.+)", line)
            if not m:
                continue
            for tech in WEAK + MATURE:
                if tech in m.group(2):
                    items.append({"name_ru": tech, "name_en": tech, "doc_ids": [int(m.group(1))]})
        return json.dumps({"items": items}, ensure_ascii=False)
    if "аналитик банка" in system:
        ids = re.findall(r"^\[([0-9a-f]{12})\]", user, flags=re.M)
        return json.dumps({"description": "Синтетическое описание для разработки интерфейса.",
                           "advantage": "Синтетическое преимущество.",
                           "case": {"text": "Синтетический кейс.", "source_id": ids[0] if ids else None},
                           "companies": ["Demo Corp"], "analyst_notes": "",
                           "summaries": {i: "Синтетическое резюме источника." for i in ids}}, ensure_ascii=False)
    if "поисковую формулировку" in system:
        name = re.search(r"«(.+?)»", user).group(1)
        return json.dumps({"name_en": name, "name_ru": name}, ensure_ascii=False)
    if "сопоставляешь" in system:
        return json.dumps({"match": False, "reason": "mock"}, ensure_ascii=False)
    return "{}"
