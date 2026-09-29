"""HTTP-клиент: ограничение частоты по хостам, повторы, кэш ответов в БД (сырые данные)."""
import hashlib
import json
import threading
import time
from urllib.parse import urlencode, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config, db

# Минимальный интервал между запросами к хосту, сек (правила площадок)
HOST_INTERVAL = {
    "export.arxiv.org": 3.1,
    "api.github.com": 2.1 if config.GITHUB_TOKEN else 6.5,  # Search API: 30/мин с токеном, 10/мин без него
    "news.google.com": 1.0,
    "api.openalex.org": 0.12,
    "en.wikipedia.org": 0.1,
    "ru.wikipedia.org": 0.1,
}
DEFAULT_INTERVAL = 0.2
USER_AGENT = "AurumForesightBot/1.0 (hackathon; contact via CONTACT_EMAIL)"

_session = requests.Session()
_retry = Retry(total=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
_session.mount("https://", HTTPAdapter(max_retries=_retry, pool_maxsize=32))
_session.mount("http://", HTTPAdapter(max_retries=_retry, pool_maxsize=32))
_session.headers["User-Agent"] = USER_AGENT

_locks: dict = {}
_last_call: dict = {}
_guard = threading.Lock()


class FetchError(Exception):
    pass


def _wait_turn(host: str):
    with _guard:
        lock = _locks.setdefault(host, threading.Lock())
    with lock:
        interval = HOST_INTERVAL.get(host, DEFAULT_INTERVAL)
        delta = time.monotonic() - _last_call.get(host, 0.0)
        if delta < interval:
            time.sleep(interval - delta)
        _last_call[host] = time.monotonic()


def get(url: str, params: dict | None = None, headers: dict | None = None, ttl_hours: float | None = 24,
        timeout: int = 25) -> str:
    """GET с кэшем. ttl_hours=None — кэш бессрочный (обучающая выборка)."""
    full = url + ("?" + urlencode(params, doseq=True) if params else "")
    key = hashlib.sha1(full.encode("utf-8")).hexdigest()
    cached = db.cache_get(key, ttl_hours)
    if cached is not None:
        if cached["status"] >= 400:
            raise FetchError(f"{cached['status']} {full}")
        return cached["body"]
    host = urlparse(url).netloc
    for attempt in range(3):
        _wait_turn(host)
        try:
            resp = _session.get(url, params=params, headers=headers or {}, timeout=timeout)
        except requests.RequestException as exc:
            raise FetchError(f"{type(exc).__name__}: {full}") from exc
        if host == "api.github.com" and resp.status_code == 403 and attempt < 2:
            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset = resp.headers.get("X-RateLimit-Reset")
            retry_after = resp.headers.get("Retry-After")
            if remaining == "0" and reset:
                delay = max(1.0, min(65.0, float(reset) - time.time() + 2.0))
            elif retry_after:
                delay = max(1.0, min(65.0, float(retry_after)))
            else:
                break
            time.sleep(delay)
            continue
        break
    body = resp.text
    # Кэшируем успех и «постоянные» 404/410. Лимиты (403/429) и сбои (5xx) не кэшируем:
    # иначе временная ошибка навсегда обнулит признак.
    if resp.status_code < 400 or resp.status_code in (404, 410):
        db.cache_put(key, full, resp.status_code, body)
    if resp.status_code >= 400:
        raise FetchError(f"{resp.status_code} {full}")
    return body


def get_json(url: str, params: dict | None = None, headers: dict | None = None, ttl_hours: float | None = 24):
    return json.loads(get(url, params=params, headers=headers, ttl_hours=ttl_hours))
