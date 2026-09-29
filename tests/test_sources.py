"""Разбор ответов внешних источников на фикстурах в реальных форматах и гонка потоков при записи в кэш."""
import os
import threading
import time

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_ws.db")
os.environ.setdefault("SOURCES_MODE", "mock")

from app import db, features, http, sources  # noqa: E402

db.init_db()

OPENALEX_GROUPS = {"meta": {"count": 200}, "results": [], "group_by": [
    {"key": "2025", "key_display_name": "2025", "count": 120}, {"key": "2024", "key_display_name": "2024", "count": 90},
    {"key": "unknown", "key_display_name": "unknown", "count": 3}]}
OPENALEX_WORKS = {"results": [
    {"id": "https://openalex.org/W1", "display_name": "Agent identity for LLMs", "publication_date": "2025-10-01",
     "language": "en", "doi": "https://doi.org/10.1/x",
     "primary_location": {"landing_page_url": "https://arxiv.org/abs/2510.00001", "source": {"display_name": "arXiv"}},
     "authorships": [{"institutions": [{"display_name": "MIT"}]}, {"institutions": []}]},
    {"id": "https://openalex.org/W2", "display_name": "No location", "publication_date": "2025-09-01",
     "language": None, "doi": None, "primary_location": None, "authorships": []}]}
GNEWS_RSS = """<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>q</title>
<item><title>Startup raises $10M for agent identity - TechCrunch</title>
<link>https://news.google.com/rss/articles/CBMiabc?oc=5</link><pubDate>Tue, 12 Aug 2025 07:00:00 GMT</pubDate>
<description>x</description><source url="https://techcrunch.com">TechCrunch</source></item>
<item><title>Пресс-релиз без источника</title><link>https://news.google.com/rss/articles/CBMidef?oc=5</link>
<pubDate>garbage date</pubDate></item></channel></rss>"""
ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2510.00001v1</id><published>2025-10-01T12:00:00Z</published>
<title>Agent
  identity   management</title><summary>We study   agents.</summary></entry></feed>"""
WIKI_SEARCH = {"query": {"search": [{"title": "Web application firewall"}, {"title": "Firewall"}]}}
WIKI_REV = {"query": {"pages": {"123": {"title": "Web application firewall", "revisions": [{"timestamp": "2009-03-04T10:11:12Z"}]}}}}


def test_openalex_parsing(monkeypatch):
    monkeypatch.setattr(http, "get_json", lambda url, params=None, **kw: OPENALEX_GROUPS if "group_by" in (params or {}) else OPENALEX_WORKS)
    assert sources.openalex_counts("x") == {2025: 120, 2024: 90}      # «unknown» отброшен
    docs = sources.openalex_recent("x")
    assert docs[0].type == "научная публикация" and docs[0].orgs == ["MIT"] and docs[0].date == "2025-10-01"
    assert docs[1].url and docs[1].lang in ("en", "")                  # null-поля не роняют разбор


def test_gnews_parsing(monkeypatch):
    monkeypatch.setattr(http, "get", lambda url, params=None, **kw: GNEWS_RSS)
    docs = sources.gnews("agent identity", "en")
    assert docs[0].title == "Startup raises $10M for agent identity" and docs[0].domain == "techcrunch.com"
    assert docs[0].date == "2025-08-12" and docs[0].type == "отраслевое СМИ"
    assert docs[1].date == "" and docs[1].title      # битая дата и отсутствие source не роняют разбор


def test_github_zero_results_uses_one_search_request(monkeypatch):
    """Пустая общая выдача гарантирует пустую выдачу за последний год."""
    calls = []

    def fake_get_json(url, params=None, **kw):
        calls.append((url, params))
        return {"total_count": 0, "items": []}

    monkeypatch.setattr(http, "get_json", fake_get_json)
    result = sources.github_stats("unlisted experimental technology")
    assert result == {"total": 0, "recent": 0, "top": []}
    assert len(calls) == 1


def test_training_evidence_skips_publication_list_without_changing_features(monkeypatch):
    """Список статей нужен отчёту, но не числовым признакам обучающей выборки."""
    from app import config
    monkeypatch.setattr(config, "SOURCES_MODE", "live")
    calls = []
    monkeypatch.setattr(sources, "openalex_counts", lambda *args, **kw: {2025: 4})
    monkeypatch.setattr(sources, "openalex_totals", lambda *args, **kw: {2025: 1_000_000, 2024: 1_000_000})

    def publications(*args, **kw):
        calls.append("pubs")
        return [sources.Document(source="openalex", url="https://openalex.org/W1", title="Example")]

    monkeypatch.setattr(sources, "openalex_recent", publications)
    monkeypatch.setattr(sources, "gnews", lambda *args, **kw: [])
    monkeypatch.setattr(sources, "github_stats", lambda *args, **kw: {"total": 0, "recent": 0, "top": []})
    monkeypatch.setattr(sources, "wikipedia", lambda *args, **kw: {"exists": False, "created_year": None})
    full = sources.collect_evidence("example")
    training = sources.collect_evidence("example", include_pubs=False)
    assert calls == ["pubs"]
    assert training["pubs"] == [] and training["errors"] == []
    assert features.compute(full) == features.compute(training)


def test_arxiv_and_wikipedia_parsing(monkeypatch):
    monkeypatch.setattr(http, "get", lambda url, params=None, **kw: ARXIV_ATOM)
    d = sources.arxiv_recent("agent identity")[0]
    assert d.title == "Agent identity management" and d.date == "2025-10-01" and d.trust == "A"
    monkeypatch.setattr(http, "get_json", lambda url, params=None, **kw: WIKI_REV if "revisions" in (params or {}).get("prop", "") else WIKI_SEARCH)
    w = sources.wikipedia("Web application firewall")
    assert w["exists"] and w["created_year"] == 2009


def test_features_ignore_russian_news():
    """Русские новости не должны влиять на признаки: их нет в окне «прошлый год» и негативы ищутся другими фразами."""
    base = {"pub_counts": {2025: 10}, "pub_totals": {2025: 1_000_000}, "news_recent": [], "news_prev": [], "news_ru": []}
    with_ru = dict(base, news_ru=[sources.Document(source="news", url=f"https://rbc.ru/{i}", title="Раунд seed привлекла", date="2026-09-01", lang="ru") for i in range(20)])
    a, b = features.compute(base), features.compute(with_ru)
    assert a == b


class _Resp:
    status_code, text = 200, "{}"


def test_cache_put_concurrent_same_key(monkeypatch):
    """Несколько потоков одновременно кэшируют один и тот же URL (так работает openalex_totals)."""
    def slow_get(url, params=None, headers=None, timeout=None):
        time.sleep(0.25)
        return _Resp()
    monkeypatch.setattr(http._session, "get", slow_get)
    errors = []

    def worker():
        try:
            http.get("https://api.openalex.org/works", {"group_by": "publication_year", "t": str(time.time_ns())[:0] + "same"}, ttl_hours=1)
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors


def test_github_search_waits_and_retries_rate_limit(monkeypatch):
    """Временный лимит GitHub не должен обнулять признаки отдельной технологии."""
    calls, sleeps, cached = [], [], []

    class Response:
        def __init__(self, status, headers):
            self.status_code = status
            self.headers = headers
            self.text = '{}'

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return (Response(403, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(time.time()))})
                if len(calls) == 1 else Response(200, {}))

    monkeypatch.setattr(http._session, "get", fake_get)
    monkeypatch.setattr(http.db, "cache_get", lambda *args: None)
    monkeypatch.setattr(http.db, "cache_put", lambda *args: cached.append(args))
    monkeypatch.setattr(http, "_wait_turn", lambda host: None)
    monkeypatch.setattr(http.time, "sleep", lambda seconds: sleeps.append(seconds))
    assert http.get("https://api.github.com/search/repositories", {"q": "example"}) == '{}'
    assert len(calls) == 2 and sleeps and len(cached) == 1
