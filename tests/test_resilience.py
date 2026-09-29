"""Устойчивость к сбоям внешних сервисов: лимиты и 5xx у моделей, лимит поиска GitHub, журнал попыток."""
import os
import threading
import time

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_ws.db")
os.environ.setdefault("SOURCES_MODE", "mock")

import requests  # noqa: E402

from app import config, db, http, llm  # noqa: E402

db.init_db()


class _Resp:
    def __init__(self, status, body="", headers=None):
        self.status_code, self.text, self.headers = status, body, headers or {}

    def json(self):
        import json
        return json.loads(self.text)


def _ok(model="GigaChat-2"):
    return _Resp(200, '{"model": "%s", "choices": [{"message": {"content": "готов"}}]}' % model)


def _setup_giga(monkeypatch, responses, job):
    monkeypatch.setitem(config.TASK_MODELS, "extract", "gigachat:GigaChat-2")
    monkeypatch.setattr(llm, "_giga_access_token", lambda: "token")
    monkeypatch.setattr(llm, "_giga_verify", lambda: True)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    seq = iter(responses)

    def post(*a, **kw):
        r = next(seq)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(llm.requests, "post", post)
    llm.CURRENT_JOB["id"] = job


def test_llm_retries_429_and_5xx_then_logs_each_attempt(monkeypatch):
    _setup_giga(monkeypatch, [_Resp(429, "too many", {"Retry-After": "1"}), _Resp(503, "busy"),
                              requests.Timeout("read timed out"), _ok()], "job-retry")
    try:
        assert llm.call("extract", "s", "u") == "готов"
    finally:
        llm.CURRENT_JOB["id"] = None
    log = db.llm_calls("job-retry")
    assert [r["ok"] for r in log] == [False, False, False, True]   # каждая попытка видна в журнале
    assert "429" in log[0]["error"] and "Timeout" in log[2]["error"]


def test_llm_gives_up_after_retries_and_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setattr(config, "LLM_RETRIES", 2)
    _setup_giga(monkeypatch, [_Resp(500, "x")] * 3, "job-giveup")
    try:
        llm.call("extract", "s", "u")
        raise AssertionError("ожидалась ошибка")
    except llm.LLMError as exc:
        assert "после 3 попыток" in str(exc)
    _setup_giga(monkeypatch, [_Resp(400, "bad model id")], "job-400")
    try:
        llm.call("extract", "s", "u")
        raise AssertionError("ожидалась ошибка")
    except llm.LLMError as exc:
        assert "400" in str(exc) and "bad model id" in str(exc)   # тело ответа видно: так ловится неверный идентификатор
    finally:
        llm.CURRENT_JOB["id"] = None
    assert len(db.llm_calls("job-400")) == 1


def test_llm_401_refreshes_token_and_logs_served_model(monkeypatch):
    _setup_giga(monkeypatch, [_Resp(401, "expired"), _ok("GigaChat-2:2.0.28")], "job-401")
    llm._giga_token["value"] = "old"
    try:
        assert llm.call("extract", "s", "u") == "готов"
    finally:
        llm.CURRENT_JOB["id"] = None
    assert llm._giga_token["value"] is None           # старый токен сброшен, следующий вызов получит новый
    log = db.llm_calls("job-401")
    assert log[-1]["model"] == "GigaChat-2 → GigaChat-2:2.0.28"


def test_llm_concurrency_cap(monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_CONCURRENCY", 2)
    monkeypatch.setitem(config.TASK_MODELS, "extract", "gigachat:GigaChat-2")
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def fake(model, system, user, temperature, max_tokens):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return "ok"
    monkeypatch.setitem(llm.PROVIDERS, "gigachat", fake)
    threads = [threading.Thread(target=llm.call, args=("extract", "s", "u")) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert state["peak"] == 2


def test_missing_ca_bundle_is_explicit(monkeypatch):
    monkeypatch.setenv("GIGACHAT_CA_BUNDLE", "/nonexistent/ca.pem")
    try:
        llm._giga_verify()
        raise AssertionError("ожидалась ошибка")
    except llm.LLMError as exc:
        assert "certs/README.md" in str(exc)


def test_github_rate_limit_403_waits_and_retries(monkeypatch):
    reset = str(int(time.time()) + 5)
    responses = iter([_Resp(403, '{"message": "API rate limit exceeded"}',
                            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset}),
                      _Resp(200, '{"total_count": 7}')])
    pauses = []
    monkeypatch.setattr(http._session, "get", lambda *a, **kw: next(responses))
    monkeypatch.setattr(http.time, "sleep", pauses.append)
    assert http.get_json("https://api.github.com/search/repositories", {"q": f"rl-{time.time_ns()}"}) == {"total_count": 7}
    assert pauses and 0 < pauses[-1] <= 65


def test_plain_403_is_not_retried(monkeypatch):
    calls = []

    def get(*a, **kw):
        calls.append(1)
        return _Resp(403, "forbidden")
    monkeypatch.setattr(http._session, "get", get)
    try:
        http.get("https://example.org/private", {"q": str(time.time_ns())})
        raise AssertionError("ожидалась ошибка")
    except http.FetchError:
        pass
    assert len(calls) == 1
