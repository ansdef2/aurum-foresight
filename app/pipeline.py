"""Сквозной конвейер одного запроса: запрос → источники → кандидаты → детектор → ТОП-15 → отчёты."""
import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import candidates, config, db, detector, features, llm, report, sources
from .detector import STAGE_LABELS, TREND_LABELS

STAGES = ["Разбор запроса", "Сбор источников", "Выделение кандидатов", "Проверка кандидатов", "Отчёты по сигналам"]


def _label(p: float, model: dict) -> str:
    return "High" if p >= config.CONFIDENT else "Med" if detector.is_weak_signal(p, model) else "Low"


def run(job_id: str, query: str) -> dict:
    llm.CURRENT_JOB["id"] = job_id
    t0 = time.monotonic()
    timings, notes = {}, []
    mark = {"t": t0}

    def stage(i: int, progress: float):
        now = time.monotonic()
        if i > 0:
            timings[STAGES[i - 1]] = round(now - mark["t"], 1)
        mark["t"] = now
        db.update_job(job_id, status="running", stage=STAGES[i], progress=progress)

    # 1. Разбор запроса
    stage(0, 0.03)
    interp = candidates.expand_query(query)

    # 2. Сбор источников
    stage(1, 0.08)
    docs, errors = candidates.discover(interp["subtopics"])
    notes += [f"Источник недоступен: {e}" for e in errors[:5]]
    db.save_documents(job_id, docs)
    docs_by_id = {d.id: d for d in docs}

    # 3. Кандидаты
    stage(2, 0.35)
    items = candidates.extract(query, docs)
    cands, rejected = candidates.cluster(items, docs_by_id)
    cands = cands[: config.MAX_CANDIDATES]

    # 4. Проверка кандидатов детектором
    stage(3, 0.5)
    model = detector.load_model()
    evaluated, evidence_docs_total = [], 0
    budget_verify = config.TIME_BUDGET_SEC * 0.7
    with ThreadPoolExecutor(config.WORKERS) as pool:
        futs = {pool.submit(sources.collect_evidence, c["name_en"], c["name_ru"]): c for c in cands}
        done = 0
        for fut in as_completed(futs):
            c = futs[fut]
            done += 1
            db.update_job(job_id, progress=0.5 + 0.3 * done / max(1, len(futs)))
            try:
                ev = fut.result()
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Кандидат «{c['name_ru']}» не проверен: {exc}"[:200])
                continue
            n_ev = len(ev.get("news_recent") or []) + len(ev.get("news_ru") or []) + len(ev.get("pubs") or [])
            evidence_docs_total += n_ev + len(ev.get("news_prev") or [])
            f = features.compute(ev)
            p, contrib = detector.gate(f, model)
            st, tr = detector.stage_trend(f, model)
            evaluated.append({"c": c, "ev": ev, "f": f, "p": p, "contrib": contrib, "stage": st, "trend": tr,
                              "n_ev": n_ev})
            if time.monotonic() - t0 > budget_verify:
                notes.append("Проверка остановлена по бюджету времени; часть кандидатов не оценена")
                for other in futs:
                    other.cancel()
                break

    accepted = []
    for e in evaluated:
        srcs = report.pick_sources(e["c"]["doc_ids"], docs_by_id, e["ev"])
        e["sources"] = srcs
        low_trust = bool(srcs) and all(d.trust == "D" for d in srcs)
        if low_trust:
            e["p"] *= 0.7
        e["flags"] = ["только источники пониженной доверенности — нужна независимая проверка"] if low_trust else []
        if detector.is_weak_signal(e["p"], model) and e["stage"] <= 4:
            accepted.append(e)
        else:
            category, reason = detector.rejection(e["f"], e["contrib"], e["n_ev"])
            if e["stage"] == 5 and detector.is_weak_signal(e["p"], model):
                category, reason = "зрелая технология", "детектор стадии относит к массовому внедрению"
            rejected.append({"name_ru": e["c"]["name_ru"], "name_en": e["c"]["name_en"], "category": category,
                             "reason": reason, "confidence": round(e["p"], 3)})
    accepted.sort(key=lambda e: e["p"], reverse=True)
    top = accepted[: config.TOP_N]

    # 5. Отчёты по ТОП-15
    stage(4, 0.82)
    with ThreadPoolExecutor(3) as pool:
        reports = list(pool.map(lambda e: report.build(e["c"], e["sources"]), top))

    signals = []
    for rank, (e, rep) in enumerate(zip(top, reports), start=1):
        expl = detector.explain(e["f"], e["contrib"])
        signals.append({
            "rank": rank, "name_ru": e["c"]["name_ru"], "name_en": e["c"]["name_en"], "aliases": e["c"]["aliases"],
            "area": interp["theme_ru"], "confidence": round(e["p"], 3), "confidence_label": _label(e["p"], model),
            "stage": {"code": e["stage"], "label": STAGE_LABELS[e["stage"]]},
            "trend": {"code": e["trend"], "label": TREND_LABELS[e["trend"]]},
            "score": e["stage"] + e["trend"],
            "key_predictors": [x["text"] for x in expl["pros"][:2]],
            "why_weak": report.why_weak(e["stage"], e["trend"], expl, e["p"]),
            "explanation": expl, "ladder": features.ladder(e["f"]),
            "features": {k: round(v, 4) for k, v in e["f"].items()},
            "flags": e["flags"] + (["источник данных с ошибками: " + "; ".join(e["ev"]["errors"])[:200]]
                                   if e["ev"].get("errors") else []),
            **rep,
        })

    top_ids, acc_ids = {id(e) for e in top}, {id(e) for e in accepted}
    all_candidates = [{"name_ru": e["c"]["name_ru"], "name_en": e["c"]["name_en"], "confidence": round(e["p"], 3),
                       "status": "в ТОП-15" if id(e) in top_ids else ("слабый сигнал" if id(e) in acc_ids
                                                                     else "отклонён"),
                       "support": e["c"]["support"]} for e in sorted(evaluated, key=lambda e: -e["p"])]
    timings[STAGES[-1]] = round(time.monotonic() - mark["t"], 1)
    timings["Всего"] = round(time.monotonic() - t0, 1)
    return {
        "query": query, "interpreted": interp,
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "models": dict(config.TASK_MODELS), "detector": {"source": model.get("source"),
                                                         "trained_at": model.get("trained_at")},
        "timings": timings, "notes": notes,
        "stats": {"sources_processed": len(docs) + evidence_docs_total, "documents_discovery": len(docs),
                  "candidates": len(evaluated), "weak_signals": len(accepted),
                  "confident": sum(1 for e in accepted if e["p"] >= config.CONFIDENT),
                  "rejected": len(rejected)},
        "signals": signals, "candidates": all_candidates, "rejected": rejected,
    }


def run_job(job_id: str, query: str):
    try:
        result = run(job_id, query)
        db.update_job(job_id, status="done", stage="Готово", progress=1.0, result=result,
                      timings=result["timings"], finished_at=db.now())
    except Exception as exc:  # noqa: BLE001
        db.update_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:1000], finished_at=db.now())
    finally:
        llm.CURRENT_JOB["id"] = None
