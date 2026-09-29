"""Тесты ядра: математика признаков, детектор, сквозной конвейер и API в офлайн-режиме."""
import os

import pytest
import time

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_ws.db")
os.environ["SOURCES_MODE"] = "mock"
for task in ("EXPAND", "EXTRACT", "CANONICAL", "REPORT", "JUDGE"):
    os.environ[f"MODEL_{task}"] = "mock:mock"
os.environ["MODEL_PATH"] = "data/model/_absent.json"

from app import db, detector, features, mock  # noqa: E402

db.init_db()


def test_gamma_poisson_small_counts():
    assert features.gamma_poisson_prob(30, 1, 3, 1) > 0.95        # явный рост
    assert features.gamma_poisson_prob(3, 1, 3, 1) < 0.4          # ровно
    assert 0.2 < features.gamma_poisson_prob(3, 1, 1, 1) < 0.9    # 1→3: неуверенно, не «+200%»


def test_features_separate_mature_and_weak():
    fw = features.compute(mock.evidence("kv cache offloading"))
    fm = features.compute(mock.evidence("contactless payments"))
    assert fm["wiki_age"] > 10 and fw["wiki_age"] == 0
    assert fm["news_saturated"] == 1 and fw["news_saturated"] == 0
    pw, _ = detector.gate(fw)
    pm, _ = detector.gate(fm)
    assert pw > 0.5 > pm


def test_training_roundtrip():
    rows = []
    for i, name in enumerate(mock.WEAK + mock.MATURE * 3):
        f = features.compute(mock.evidence(name))
        weak = name in mock.WEAK
        rows.append({**f, "label": int(weak), "area": f"t{i % 3}", "stage": 2 + i % 3 if weak else 5,
                     "trend": 3 if i % 2 else 2})
    m = detector.train(rows)
    assert m["gate_threshold"] == detector.GATE_THRESHOLD
    assert m["metrics"]["gate"]["accuracy"] > 0.8
    p, contrib = detector.gate(features.compute(mock.evidence("orbital edge inference")), m)
    assert set(contrib) == set(features.FEATURES)
    detector.stage_trend(features.compute(mock.evidence("orbital edge inference")), m)


def test_gate_decision_uses_model_threshold():
    model = {"gate_threshold": 0.64}
    assert not detector.is_weak_signal(0.63, model)
    assert detector.is_weak_signal(0.64, model)
    assert detector.is_weak_signal(0.65, model)


def test_gate_fit_respects_methodology_signs():
    import numpy as np

    names = ["mainstream_share", "news_growth_p"]
    X = np.array([[0.0, 0.1], [0.1, 0.1], [0.2, 0.1], [0.8, 0.9], [0.9, 0.9], [1.0, 0.9]])
    y = np.array([0, 0, 0, 1, 1, 1])
    gate = detector._fit_block(X, y, names, C=0.05, sign_constraints=True)
    assert gate["coef"][0] < 0
    assert gate["coef"][1] > 0


def test_api_end_to_end():
    from fastapi.testclient import TestClient
    from app.api import app
    with TestClient(app) as client:
        job_id = client.post("/api/search", json={"query": "Защита ИИ", "force": True}).json()["job_id"]
        for _ in range(100):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(0.1)
        assert job["status"] == "done", job.get("error")
        res = job["result"]
        assert 0 < len(res["signals"]) <= 15
        s = res["signals"][0]
        for key in ("name_ru", "confidence", "why_weak", "description", "advantage", "case", "sources"):
            assert key in s
        assert all({"title", "url", "date", "type", "lang", "trust"} <= set(x) for x in s["sources"])
        assert any(r["category"] == "зрелая технология" for r in res["rejected"])
        assert client.get(f"/api/jobs/{job_id}/export.xlsx").status_code == 200
        assert len(client.get(f"/api/jobs/{job_id}/llm").json()) > 0
        assert client.get("/api/model").json()["source"] == "prior"


def test_no_dataset_leak_into_app():
    """Пакет app/ не должен читать эталонный датасет: поиск обязан работать без него."""
    import pathlib
    for path in pathlib.Path("app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "data/eval" not in text and "dataset.xlsx" not in text and "scripts.dataset" not in text, path


@pytest.mark.skipif(not os.path.exists("data/eval/dataset.xlsx"), reason="датасет заказчика не входит в репозиторий")
def test_training_scripts_mock(tmp_path, monkeypatch):
    """Сборка признаков и обучение отрабатывают на офлайн-данных."""
    import pandas as pd
    from scripts.dataset import load
    df = load()
    assert len(df) == 100 and ((df.stage + df.trend) == df.score).mean() >= 0.85
    neg = pd.read_csv("data/negatives.csv")
    assert set(neg.area) == set(df.area) and len(neg) >= 90


def test_kleinberg_burst_automaton():
    flat = features.burst_states([3] * 24)
    rise = features.burst_states([1] * 18 + [9, 10, 12, 11, 10, 12])
    spike = features.burst_states([1] * 23 + [3])
    assert sum(flat) == 0                       # ровный поток — не всплеск
    assert sum(rise[-6:]) == 6 and sum(rise[:18]) == 0   # устойчивый подъём — всплеск ровно на своих месяцах
    assert sum(spike) == 0                      # единичный скачок — не всплеск
    assert features.burst_states([]) == [] and features.burst_states([0] * 6) == [0] * 6


def test_monthly_counts_and_pinned_date(monkeypatch):
    import datetime as dt
    from app import config
    counts = features.monthly_counts(["2026-09-01", "2026-08-15", "2024-10-03", "2020-01-01", ""], dt.date(2026, 9, 28))
    assert len(counts) == 24 and counts[-1] == 1 and counts[-2] == 1 and counts[0] == 1 and sum(counts) == 3
    monkeypatch.setattr(config, "AS_OF_DATE", "2026-01-15")
    assert config.today() == dt.date(2026, 1, 15)
