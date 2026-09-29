"""HTTP API и раздача интерфейса."""
import io
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, db, detector, pipeline

app = FastAPI(title="Aurum Foresight System", version="1.0",
              description="Поиск слабых технологических сигналов в открытых источниках")
_executor = ThreadPoolExecutor(max_workers=1)  # запросы выполняются по очереди: лимиты API и честные замеры
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")  # логотип, шрифты, React — без внешних CDN


@app.on_event("startup")
def _startup():
    db.init_db()
    db.fail_stale_jobs()


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    force: bool = False


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


@app.post("/api/search")
def search(body: SearchIn):
    qn = _norm(body.query)
    if not body.force:
        cached = db.find_cached_job(qn, config.CACHE_TTL_HOURS)
        if cached:
            return {"job_id": cached, "cached": True}
    job_id = uuid.uuid4().hex[:16]
    db.create_job(job_id, body.query.strip(), qn)
    _executor.submit(pipeline.run_job, job_id, body.query.strip())
    return {"job_id": job_id, "cached": False}


@app.get("/api/jobs")
def jobs():
    return db.list_jobs()


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404, "Запрос не найден")
    return j


@app.get("/api/jobs/{job_id}/llm")
def job_llm(job_id: str):
    return db.llm_calls(job_id)


@app.get("/api/model")
def model():
    m = detector.load_model()
    g = m["gate"]
    from .features import LABELS
    weights = sorted(({"feature": f, "label": LABELS[f], "weight": round(w, 3)} for f, w in zip(m["features"], g["coef"])),
                     key=lambda x: -abs(x["weight"]))
    return {"source": m.get("source"), "trained_at": m.get("trained_at"), "weights": weights,
            "gate_threshold": m.get("gate_threshold", 0.5), "metrics": m.get("metrics"), "models": config.TASK_MODELS}


@app.get("/api/health")
def health():
    return {"ok": True, "sources_mode": config.SOURCES_MODE, "detector": detector.load_model().get("source")}


DATASET_COLUMNS = ["№", "Технология (слабый сигнал)", "Область", "Компании", "Почему это слабый сигнал",
                   "Стадия развития", "Тренд упоминаний", "Балл (стадия+тренд)", "Источники",
                   "Уверенность модели", "Описание", "Преимущество", "Кейс-пример"]


@app.get("/api/jobs/{job_id}/export.xlsx")
def export(job_id: str):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    j = db.get_job(job_id)
    if not j or not j.get("result"):
        raise HTTPException(404, "Результат ещё не готов")
    res = j["result"]
    wb = Workbook()
    ws = wb.active
    ws.title = "Слабые сигналы"
    ws.append([f"Aurum Foresight System · запрос: {res['query']} · сформировано {res['generated_at']}"])
    ws.append(DATASET_COLUMNS)
    for s in res["signals"]:
        links = ", ".join(f"[{x['title'][:80]}]({x['url']})" for x in s["sources"])
        ws.append([s["rank"], s["name_ru"], s["area"], ", ".join(s.get("companies") or []), s["why_weak"],
                   s["stage"]["label"], s["trend"]["label"], s["score"], links, s["confidence"],
                   s.get("description", ""), s.get("advantage", ""), (s.get("case") or {}).get("text", "")])
    for row in ws.iter_rows():
        for cell in row:
            cell.font = Font(name="Arial", size=10, bold=cell.row == 2)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for col, width in zip("ABCDEFGHIJKLM", [5, 40, 16, 28, 60, 20, 16, 10, 60, 12, 50, 40, 40]):
        ws.column_dimensions[col].width = width
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": f'attachment; filename="weak_signals_{job_id}.xlsx"'})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC / "logo.svg", media_type="image/svg+xml")
