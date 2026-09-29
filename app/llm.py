"""Клиенты разрешённых моделей. Каждый вызов пишется в журнал llm_log (модель, задача, время).

LLM здесь только извлекает факты, переводит и пишет тексты. Решение «слабый сигнал или нет»
принимает детектор (app/detector.py), а не языковая модель.
"""
import json
import os
import random
import re
import threading
import time
import uuid

import requests

from . import config, db

CURRENT_JOB = {"id": None}  # задачи выполняются по одной, поэтому достаточно глобального значения
RETRY_STATUS = (429, 500, 502, 503, 504)


class LLMError(Exception):
    pass


class RetryableError(LLMError):
    """Временный сбой провайдера (лимит, 5xx, таймаут): вызов повторяется с паузой."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _check(resp: requests.Response, provider: str):
    """Ошибки HTTP: лимиты и 5xx — повторяемые, остальные — сразу наверх с телом ответа."""
    if resp.status_code < 400:
        return
    body = (resp.text or "")[:300]
    if resp.status_code in RETRY_STATUS:
        ra = resp.headers.get("Retry-After")
        raise RetryableError(f"{provider} HTTP {resp.status_code}: {body}",
                             float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
    raise LLMError(f"{provider} HTTP {resp.status_code}: {body}")


def model_for(task: str) -> tuple[str, str]:
    spec = config.TASK_MODELS[task]
    provider, model = spec.split(":", 1)
    allowed = config.ALLOWED_MODELS.get(provider, set())
    if provider not in config.ALLOWED_MODELS:
        raise LLMError(f"Провайдер {provider} не входит в разрешённые ТЗ")
    if allowed is not None and model.split("/")[0] not in allowed:
        raise LLMError(f"Модель {model} не входит в разрешённые ТЗ")
    return provider, model


# ---------------------------------------------------------------- GigaChat
_giga_token = {"value": None, "expires": 0.0}
_giga_lock = threading.Lock()


def _giga_verify():
    bundle = config.env("GIGACHAT_CA_BUNDLE")
    if bundle:
        if not os.path.exists(bundle):
            raise LLMError(f"Нет файла сертификата GIGACHAT_CA_BUNDLE={bundle} (см. certs/README.md)")
        return bundle
    return config.env("GIGACHAT_VERIFY_SSL", "true").lower() != "false"


def _giga_access_token() -> str:
    with _giga_lock:
        if _giga_token["value"] and time.time() < _giga_token["expires"] - 60:
            return _giga_token["value"]
        auth_key = config.env("GIGACHAT_AUTH_KEY")
        if not auth_key:
            raise LLMError("Не задан GIGACHAT_AUTH_KEY")
        resp = requests.post(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            headers={"Authorization": f"Basic {auth_key}", "RqUID": str(uuid.uuid4()),
                     "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            data={"scope": config.env("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")},
            verify=_giga_verify(), timeout=30)
        _check(resp, "GigaChat OAuth")
        data = resp.json()
        _giga_token["value"] = data["access_token"]
        _giga_token["expires"] = data.get("expires_at", 0) / 1000 or time.time() + 1500
        return _giga_token["value"]


GIGACHAT_API = "https://gigachat.devices.sberbank.ru/api/v1"
_served = threading.local()


def _call_gigachat(model, system, user, temperature, max_tokens):
    resp = requests.post(
        f"{GIGACHAT_API}/chat/completions",
        headers={"Authorization": f"Bearer {_giga_access_token()}", "Content-Type": "application/json"},
        json={"model": model, "temperature": temperature, "max_tokens": max_tokens,
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        verify=_giga_verify(), timeout=config.LLM_TIMEOUT_SEC)
    if resp.status_code == 401:  # токен отозван или истёк раньше срока — получить новый и повторить
        with _giga_lock:
            _giga_token["value"] = None
        raise RetryableError("GigaChat HTTP 401: токен доступа истёк")
    _check(resp, "GigaChat")
    data = resp.json()
    _served.value = data.get("model")  # какая модель фактически ответила — пишется в журнал рядом с запрошенной
    return data["choices"][0]["message"]["content"]


def gigachat_models() -> list[str]:
    """Идентификаторы моделей, доступных ключу. Нужны, чтобы сверить MODEL_* перед прогоном."""
    resp = requests.get(f"{GIGACHAT_API}/models", headers={"Authorization": f"Bearer {_giga_access_token()}"},
                        verify=_giga_verify(), timeout=30)
    _check(resp, "GigaChat")
    return sorted(m.get("id", "") for m in resp.json().get("data", []))


# ---------------------------------------------------------------- YandexGPT
def _call_yandex(model, system, user, temperature, max_tokens):
    key, folder = config.env("YANDEX_API_KEY"), config.env("YANDEX_FOLDER_ID")
    if not key or not folder:
        raise LLMError("Не заданы YANDEX_API_KEY / YANDEX_FOLDER_ID")
    uri = model if model.startswith("gpt://") else f"gpt://{folder}/{model if '/' in model else model + '/latest'}"
    resp = requests.post(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        headers={"Authorization": f"Api-Key {key}", "x-folder-id": folder},
        json={"modelUri": uri,
              "completionOptions": {"stream": False, "temperature": temperature, "maxTokens": str(max_tokens)},
              "messages": [{"role": "system", "text": system}, {"role": "user", "text": user}]},
        timeout=config.LLM_TIMEOUT_SEC)
    _check(resp, "YandexGPT")
    return resp.json()["result"]["alternatives"][0]["message"]["text"]


# ---------------------------------------------------------------- OpenAI-совместимые (gpt-4.1, Qwen)
def _call_openai_compatible(base_env, key_env, default_base):
    def _call(model, system, user, temperature, max_tokens):
        key = config.env(key_env)
        if not key:
            raise LLMError(f"Не задан {key_env}")
        base = config.env(base_env, default_base).rstrip("/")
        resp = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "temperature": temperature, "max_tokens": max_tokens,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=config.LLM_TIMEOUT_SEC)
        _check(resp, key_env.split("_")[0])
        return resp.json()["choices"][0]["message"]["content"]
    return _call


def _call_mock(model, system, user, temperature, max_tokens):
    from . import mock
    return mock.llm_response(system, user)


PROVIDERS = {
    "gigachat": _call_gigachat,
    "yandex": _call_yandex,
    "openai": _call_openai_compatible("OPENAI_BASE_URL", "OPENAI_API_KEY", "https://api.openai.com/v1"),
    "qwen": _call_openai_compatible("QWEN_BASE_URL", "QWEN_API_KEY",
                                    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    "mock": _call_mock,
}


_slots = {"n": None, "sem": None}
_slots_lock = threading.Lock()


def _slot():
    """Потолок одновременных запросов к моделям (LLM_MAX_CONCURRENCY, 0 — без потолка)."""
    with _slots_lock:
        if _slots["n"] != config.LLM_MAX_CONCURRENCY:
            n = config.LLM_MAX_CONCURRENCY
            _slots["n"], _slots["sem"] = n, (threading.BoundedSemaphore(n) if n > 0 else None)
        return _slots["sem"]


def _backoff(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(retry_after, 60.0)
    return min(2.0 ** attempt + random.uniform(0, 1), 60.0)


def _call_once(task, provider, model, system, user, temperature, max_tokens) -> str:
    """Один HTTP-вызов модели — одна строка в журнале llm_log, включая неудачные попытки."""
    started, t0 = db.now(), time.monotonic()
    text, error = "", None
    _served.value = None
    sem = _slot() if provider != "mock" else None
    try:
        if sem:
            sem.acquire()
        try:
            text = PROVIDERS[provider](model, system, user, temperature, max_tokens)
        finally:
            if sem:
                sem.release()
        return text
    except LLMError as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
        raise
    except (requests.Timeout, requests.ConnectionError) as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
        raise RetryableError(error) from exc
    except Exception as exc:  # noqa: BLE001 — логируем любую ошибку провайдера
        error = f"{type(exc).__name__}: {exc}"[:500]
        raise LLMError(error) from exc
    finally:
        served = getattr(_served, "value", None)
        logged = model if not served or served == model else f"{model} → {served}"
        db.log_llm(job_id=CURRENT_JOB["id"], task=task, provider=provider, model=logged[:64], started_at=started,
                   duration_ms=int((time.monotonic() - t0) * 1000), prompt_chars=len(system) + len(user),
                   response_chars=len(text or ""), ok=error is None, error=error)


def call(task: str, system: str, user: str, temperature: float = 0.2, max_tokens: int = 2000) -> str:
    provider, model = model_for(task)
    attempt = 0
    while True:
        try:
            return _call_once(task, provider, model, system, user, temperature, max_tokens)
        except RetryableError as exc:
            if attempt >= config.LLM_RETRIES:
                raise LLMError(f"{exc} (после {attempt + 1} попыток)") from exc
            time.sleep(_backoff(attempt, exc.retry_after))
            attempt += 1


def parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError("Модель вернула не JSON")


def call_json(task: str, system: str, user: str, **kwargs):
    system = system + "\nОтвечай только валидным JSON без пояснений и без markdown."
    try:
        return parse_json(call(task, system, user, **kwargs))
    except LLMError:
        return parse_json(call(task, system, user + "\n\nВерни строго JSON.", **kwargs))
