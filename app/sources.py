"""Сборщики открытых источников и классификация источников (тип, язык, уровень доверенности).

Источники: OpenAlex (научные публикации, включая препринты), arXiv, Google News RSS (новости на
английском и русском), GitHub (репозитории), Википедия (маркер зрелости).
"""
import datetime as dt
import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from . import config, http

# ---------------------------------------------------------------- классификация доменов
PR_WIRES = {"prnewswire.com", "businesswire.com", "globenewswire.com", "accesswire.com", "einpresswire.com",
            "newswire.ca", "openpr.com", "pulse2.com", "stocktitan.net", "f6s.com", "seedlist.com",
            "trysignalbase.com", "tamradar.com", "startup-seeker.com", "jobswithstartups.com", "prweb.com",
            "newsfilecorp.com", "markets.businessinsider.com", "finance.yahoo.com"}
SOCIAL_BLOGS = {"medium.com", "substack.com", "habr.com", "vc.ru", "dzen.ru", "reddit.com", "x.com",
                "twitter.com", "linkedin.com", "t.me", "youtube.com", "facebook.com", "pikabu.ru",
                "quora.com", "hackernoon.com", "dev.to"}
MAINSTREAM = {"reuters.com", "bloomberg.com", "ft.com", "wsj.com", "nytimes.com", "bbc.com", "bbc.co.uk",
              "cnn.com", "cnbc.com", "forbes.com", "theguardian.com", "apnews.com", "economist.com",
              "washingtonpost.com", "rbc.ru", "kommersant.ru", "vedomosti.ru", "interfax.ru", "tass.ru",
              "ria.ru", "lenta.ru", "gazeta.ru", "iz.ru", "businessinsider.com", "time.com"}
TRADE_MEDIA = {"techcrunch.com", "siliconangle.com", "theverge.com", "wired.com", "arstechnica.com",
               "venturebeat.com", "zdnet.com", "theregister.com", "eetimes.com", "datacenterdynamics.com",
               "tomshardware.com", "therobotreport.com", "roboticsandautomationnews.com", "securityweek.com",
               "darkreading.com", "theblock.co", "coindesk.com", "finextra.com", "fintech.global",
               "eu-startups.com", "sifted.eu", "geekwire.com", "nextplatform.com", "semiwiki.com",
               "spectrum.ieee.org", "technologyreview.com", "theaiinsider.tech", "cnews.ru", "tadviser.ru",
               "3dnews.ru", "iksmedia.ru", "comnews.ru", "habr.com/ru/news", "fierce-network.com",
               "biometricupdate.com", "pymnts.com", "americanbanker.com", "bankinfosecurity.com",
               "hpcwire.com", "servethehome.com", "anandtech.com", "electronicsweekly.com", "edge-ai-vision.com",
               "pandaily.com", "thedeeptech.in", "techfundingnews.com", "thesaasnews.com", "runtimewire.com"}
SCIENCE = {"arxiv.org", "nature.com", "science.org", "ieee.org", "ieeexplore.ieee.org", "acm.org",
           "dl.acm.org", "springer.com", "link.springer.com", "sciencedirect.com", "mdpi.com",
           "frontiersin.org", "cyberleninka.ru", "elibrary.ru", "openreview.net", "aclanthology.org",
           "biorxiv.org", "medrxiv.org", "pnas.org", "cell.com", "wiley.com", "tandfonline.com"}
GOV_INTL = {"europa.eu", "oecd.org", "bis.org", "imf.org", "worldbank.org", "un.org", "unesco.org",
            "nist.gov", "iso.org", "cbr.ru", "fatf-gafi.org", "weforum.org", "itu.int", "ietf.org",
            "owasp.org", "enisa.europa.eu", "government.ru", "minobrnauki.gov.ru", "fips.ru"}


def domain_of(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    return host[4:] if host.startswith("www.") else host


def _match(domain: str, pool: set) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in pool)


def classify(domain: str, source: str, url: str = "") -> tuple[str, str]:
    """Возвращает (тип источника по-русски, уровень доверенности A–D)."""
    if source == "openalex":
        return "научная публикация", "A"
    if source == "arxiv":
        return "препринт", "A"
    if source == "github":
        return "репозиторий кода", "B"
    if source == "wikipedia":
        return "энциклопедия", "B"
    if _match(domain, SCIENCE):
        return "научная публикация", "A"
    if domain.endswith(".gov") or domain.endswith(".gov.ru") or domain.endswith(".edu") or _match(domain, GOV_INTL):
        return "государственный / международный / университет", "A"
    if _match(domain, PR_WIRES) or "press-release" in url or "/news-releases/" in url:
        return "пресс-релиз / агрегатор", "D"
    if _match(domain, SOCIAL_BLOGS):
        return "блог / соцсеть", "D"
    if _match(domain, MAINSTREAM):
        return "массовое СМИ", "B"
    if _match(domain, TRADE_MEDIA):
        return "отраслевое СМИ", "B"
    return "сайт компании / прочее", "C"


def detect_lang(text: str) -> str:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "en"
    cyr = sum(1 for ch in letters if "а" <= ch.lower() <= "я" or ch.lower() == "ё")
    return "ru" if cyr / len(letters) > 0.3 else "en"


@dataclass
class Document:
    source: str
    url: str
    title: str
    date: str = ""            # YYYY-MM-DD
    snippet: str = ""
    lang: str = ""
    domain: str = ""
    type: str = ""
    trust: str = ""
    orgs: list = field(default_factory=list)
    id: str = ""

    def __post_init__(self):
        self.domain = self.domain or domain_of(self.url)
        self.lang = self.lang or detect_lang(self.title)
        if not self.type or not self.trust:
            self.type, self.trust = classify(self.domain, self.source, self.url)
        self.id = self.id or hashlib.sha1((self.url or self.title).encode()).hexdigest()[:12]

    def to_dict(self):
        return asdict(self)


def _today():
    return config.today()


# ---------------------------------------------------------------- OpenAlex
OPENALEX = "https://api.openalex.org/works"


def _quoted(q: str) -> str:
    q = q.replace('"', "").strip()
    return f'"{q}"' if " " in q else q


def openalex_counts(query: str, ttl=24) -> dict:
    data = http.get_json(OPENALEX, {"search": _quoted(query), "group_by": "publication_year",
                                    "mailto": config.CONTACT_EMAIL}, ttl_hours=ttl)
    return {int(g["key"]): int(g["count"]) for g in data.get("group_by", []) if str(g.get("key", "")).isdigit()}


def openalex_totals(ttl=24 * 7) -> dict:
    """Объём всего корпуса OpenAlex по годам — знаменатель для нормировки роста."""
    data = http.get_json(OPENALEX, {"group_by": "publication_year", "mailto": config.CONTACT_EMAIL}, ttl_hours=ttl)
    return {int(g["key"]): int(g["count"]) for g in data.get("group_by", []) if str(g.get("key", "")).isdigit()}


def openalex_recent(query: str, months: int = 24, per_page: int = 25, ttl=24) -> list[Document]:
    since = (_today() - dt.timedelta(days=30 * months)).isoformat()
    data = http.get_json(OPENALEX, {
        "search": _quoted(query), "filter": f"from_publication_date:{since}", "sort": "relevance_score:desc",
        "per_page": per_page, "mailto": config.CONTACT_EMAIL,
        "select": "id,display_name,publication_date,language,primary_location,doi,authorships"}, ttl_hours=ttl)
    docs = []
    for w in data.get("results", []):
        loc = w.get("primary_location") or {}
        url = loc.get("landing_page_url") or w.get("doi") or w.get("id")
        orgs = sorted({i.get("display_name") for a in (w.get("authorships") or [])
                       for i in (a.get("institutions") or []) if i.get("display_name")})
        venue = ((loc.get("source") or {}).get("display_name")) or ""
        docs.append(Document(source="openalex", url=url or "", title=w.get("display_name") or "",
                             date=w.get("publication_date") or "", snippet=venue,
                             lang=(w.get("language") or "")[:2], orgs=orgs[:8]))
    return docs


# ---------------------------------------------------------------- arXiv
def arxiv_recent(query: str, max_results: int = 30, ttl=24) -> list[Document]:
    body = http.get("https://export.arxiv.org/api/query", {
        "search_query": f"all:{_quoted(query)}", "sortBy": "submittedDate", "sortOrder": "descending",
        "max_results": max_results}, ttl_hours=ttl)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(body)
    docs = []
    for e in root.findall("a:entry", ns):
        title = " ".join((e.findtext("a:title", "", ns) or "").split())
        summary = " ".join((e.findtext("a:summary", "", ns) or "").split())
        docs.append(Document(source="arxiv", url=e.findtext("a:id", "", ns), title=title,
                             date=(e.findtext("a:published", "", ns) or "")[:10], snippet=summary[:400], lang="en"))
    return docs


# ---------------------------------------------------------------- Google News RSS
def gnews(query: str, lang: str = "en", days_from: int = 365, days_to: int = 0, ttl=12) -> list[Document]:
    """Новости за окно [сегодня − days_from, сегодня − days_to]. RSS отдаёт не больше 100 записей."""
    after = (_today() - dt.timedelta(days=days_from)).isoformat()
    before = (_today() - dt.timedelta(days=days_to)).isoformat()
    q = f"{_quoted(query)} after:{after} before:{before}"
    locale = {"en": {"hl": "en-US", "gl": "US", "ceid": "US:en"}, "ru": {"hl": "ru", "gl": "RU", "ceid": "RU:ru"}}[lang]
    body = http.get("https://news.google.com/rss/search", {"q": q, **locale}, ttl_hours=ttl)
    root = ET.fromstring(body)
    docs = []
    for item in root.iter("item"):
        title = item.findtext("title") or ""
        src = item.find("source")
        src_url = src.get("url") if src is not None else ""
        src_name = src.text if src is not None and src.text else ""
        if src_name and title.endswith(" - " + src_name):
            title = title[: -len(src_name) - 3]
        try:
            date = parsedate_to_datetime(item.findtext("pubDate") or "").date().isoformat()
        except (TypeError, ValueError):
            date = ""
        docs.append(Document(source="news", url=item.findtext("link") or "", title=title, date=date,
                             snippet=src_name, lang=lang, domain=domain_of(src_url) if src_url else ""))
    return docs


# ---------------------------------------------------------------- GitHub
def github_stats(query: str, ttl=24) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    since = (_today() - dt.timedelta(days=365)).isoformat()
    q = _quoted(query)
    total = http.get_json("https://api.github.com/search/repositories", {"q": q, "per_page": 3, "sort": "stars"},
                          headers=headers, ttl_hours=ttl)
    total_count = int(total.get("total_count", 0))
    if total_count == 0:
        return {"total": 0, "recent": 0, "top": []}
    recent = http.get_json("https://api.github.com/search/repositories",
                           {"q": f"{q} created:>={since}", "per_page": 3, "sort": "stars"}, headers=headers, ttl_hours=ttl)
    top = [Document(source="github", url=r["html_url"], title=f"{r['full_name']}: {r.get('description') or ''}"[:300],
                    date=(r.get("created_at") or "")[:10], snippet=f"★ {r.get('stargazers_count', 0)}", lang="en")
           for r in (recent.get("items") or [])[:2]]
    return {"total": total_count, "recent": int(recent.get("total_count", 0)), "top": top}


# ---------------------------------------------------------------- Википедия
def _tokens(s: str) -> set:
    return {t for t in re.findall(r"[a-zа-яё0-9]+", s.lower()) if len(t) > 2}


def wikipedia(query: str, lang: str = "en", ttl=24 * 7) -> dict:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    data = http.get_json(api, {"action": "query", "list": "search", "srsearch": query, "srlimit": 3,
                               "format": "json"}, ttl_hours=ttl)
    q = _tokens(query)
    for hit in data.get("query", {}).get("search", []):
        t = _tokens(hit["title"])
        if q and t and len(q & t) / len(q | t) >= 0.6:
            rev = http.get_json(api, {"action": "query", "prop": "revisions", "titles": hit["title"], "rvlimit": 1,
                                      "rvdir": "newer", "rvprop": "timestamp", "format": "json"}, ttl_hours=ttl)
            pages = rev.get("query", {}).get("pages", {})
            ts = next((p.get("revisions", [{}])[0].get("timestamp", "") for p in pages.values()), "")
            year = int(ts[:4]) if ts[:4].isdigit() else None
            url = f"https://{lang}.wikipedia.org/wiki/{hit['title'].replace(' ', '_')}"
            return {"exists": True, "title": hit["title"], "created_year": year, "url": url}
    return {"exists": False, "title": None, "created_year": None, "url": None}


# ---------------------------------------------------------------- доказательная база кандидата
def collect_evidence(name_en: str, name_ru: str = "", ttl=24, include_pubs: bool = True) -> dict:
    """Всё, что нужно детектору по одному кандидату. Ошибка одного источника не роняет остальные."""
    if config.SOURCES_MODE == "mock":
        from . import mock
        return mock.evidence(name_en, name_ru)
    ev = {"name_en": name_en, "name_ru": name_ru, "errors": []}

    def safe(key, fn, default):
        try:
            ev[key] = fn()
        except Exception as exc:  # noqa: BLE001 — отказоустойчивость: фиксируем и продолжаем
            ev[key] = default
            ev["errors"].append(f"{key}: {exc}"[:200])

    safe("pub_counts", lambda: openalex_counts(name_en, ttl), {})
    safe("pub_totals", lambda: openalex_totals(), {})
    if include_pubs:
        safe("pubs", lambda: openalex_recent(name_en, 24, 10, ttl), [])
    else:
        ev["pubs"] = []  # список работ не входит в числовые признаки обучения
    safe("news_recent", lambda: gnews(name_en, "en", 365, 0, ttl), [])
    safe("news_prev", lambda: gnews(name_en, "en", 730, 365, ttl), [])
    safe("news_ru", lambda: gnews(name_ru, "ru", 365, 0, ttl) if name_ru else [], [])
    safe("github", lambda: github_stats(name_en, ttl), {"total": 0, "recent": 0, "top": []})
    safe("wiki", lambda: wikipedia(name_en, "en"), {"exists": False, "created_year": None, "url": None, "title": None})
    return ev
