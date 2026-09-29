"""Хранилище: задачи, журнал вызовов моделей, сырые ответы источников, документы."""
import datetime as dt

from sqlalchemy.exc import IntegrityError
from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, Text,
                        create_engine, select, update)

from . import config

_kwargs = {"pool_pre_ping": True, "future": True}
if config.DATABASE_URL.startswith("sqlite"):
    _kwargs["connect_args"] = {"check_same_thread": False, "timeout": 60}
engine = create_engine(config.DATABASE_URL, **_kwargs)
metadata = MetaData()

jobs = Table(
    "jobs", metadata,
    Column("id", String(32), primary_key=True),
    Column("query", Text, nullable=False),
    Column("query_norm", Text, index=True),
    Column("status", String(16), nullable=False),      # queued | running | done | failed
    Column("stage", String(64)),
    Column("progress", Float, default=0.0),
    Column("created_at", DateTime),
    Column("finished_at", DateTime),
    Column("timings", JSON),
    Column("result", JSON),
    Column("error", Text),
)

llm_log = Table(
    "llm_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("job_id", String(32), index=True),
    Column("task", String(32)),
    Column("provider", String(32)),
    Column("model", String(64)),
    Column("started_at", DateTime),
    Column("duration_ms", Integer),
    Column("prompt_chars", Integer),
    Column("response_chars", Integer),
    Column("ok", Boolean),
    Column("error", Text),
)

http_cache = Table(
    "http_cache", metadata,
    Column("key", String(64), primary_key=True),
    Column("url", Text),
    Column("status", Integer),
    Column("body", Text),
    Column("fetched_at", DateTime),
)

documents = Table(
    "documents", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("job_id", String(32), index=True),
    Column("doc_id", String(64)),
    Column("source", String(32)),
    Column("url", Text),
    Column("title", Text),
    Column("date", String(10)),
    Column("lang", String(8)),
    Column("type", String(64)),
    Column("trust", String(2)),
    Column("domain", String(128)),
)


def init_db():
    metadata.create_all(engine)


def now():
    return dt.datetime.utcnow()


def create_job(job_id: str, query: str, query_norm: str):
    with engine.begin() as c:
        c.execute(jobs.insert().values(id=job_id, query=query, query_norm=query_norm, status="queued",
                                       stage="В очереди", progress=0.0, created_at=now()))


def update_job(job_id: str, **values):
    with engine.begin() as c:
        c.execute(update(jobs).where(jobs.c.id == job_id).values(**values))


def get_job(job_id: str):
    with engine.connect() as c:
        row = c.execute(select(jobs).where(jobs.c.id == job_id)).mappings().first()
        return dict(row) if row else None


def list_jobs(limit: int = 20):
    with engine.connect() as c:
        rows = c.execute(select(jobs.c.id, jobs.c.query, jobs.c.status, jobs.c.created_at, jobs.c.finished_at)
                         .order_by(jobs.c.created_at.desc()).limit(limit)).mappings().all()
        return [dict(r) for r in rows]


def find_cached_job(query_norm: str, max_age_hours: int):
    since = now() - dt.timedelta(hours=max_age_hours)
    with engine.connect() as c:
        row = c.execute(select(jobs.c.id).where(jobs.c.query_norm == query_norm, jobs.c.status == "done",
                                                jobs.c.created_at >= since)
                        .order_by(jobs.c.created_at.desc())).first()
        return row[0] if row else None


def fail_stale_jobs():
    with engine.begin() as c:
        c.execute(update(jobs).where(jobs.c.status.in_(["queued", "running"]))
                  .values(status="failed", error="Сервис перезапущен во время выполнения"))


def log_llm(**values):
    with engine.begin() as c:
        c.execute(llm_log.insert().values(**values))


def llm_calls(job_id: str):
    with engine.connect() as c:
        rows = c.execute(select(llm_log).where(llm_log.c.job_id == job_id).order_by(llm_log.c.id)).mappings().all()
        return [dict(r) for r in rows]


def cache_get(key: str, max_age_hours):
    with engine.connect() as c:
        row = c.execute(select(http_cache).where(http_cache.c.key == key)).mappings().first()
    if not row:
        return None
    if max_age_hours is not None and row["fetched_at"] < now() - dt.timedelta(hours=max_age_hours):
        return None
    return dict(row)


def cache_put(key: str, url: str, status: int, body: str):
    try:
        with engine.begin() as c:
            c.execute(http_cache.delete().where(http_cache.c.key == key))
            c.execute(http_cache.insert().values(key=key, url=url, status=status, body=body, fetched_at=now()))
    except IntegrityError:
        pass  # другой поток успел записать тот же ответ (на PostgreSQL возможно для общих URL)


def save_documents(job_id: str, docs):
    if not docs:
        return
    rows = [dict(job_id=job_id, doc_id=d.id, source=d.source, url=d.url, title=d.title[:2000], date=d.date,
                 lang=d.lang, type=d.type, trust=d.trust, domain=d.domain) for d in docs]
    with engine.begin() as c:
        c.execute(documents.insert(), rows)
