"""Handover Parquet generation for cross-district recomputation.

A handover job snapshots the two rounds' *currently published* generations at
the moment the job is created (via the same refcount mechanism as analysis
queries), then aggregates, per round and district:

* the main weight row: total weight and bus-weight sums;
* one row per replicate id: total weight and bus-weight sums.

Both rounds are written into a single Parquet file even when their replicate
id sets differ; rows whose weight sums are zero are kept as-is.  The two
generations are recorded both as columns (``generation``) and in the file
name, so receivers always know which two data vintages the file corresponds
to.

Jobs run on a single daemon worker thread.  Replacement uploads during
generation only move the ``meta_current`` pointers: pinned generations stay
readable until the job finishes, so a job always reflects the two rounds
captured when it was created.  Job metadata is committed to DuckDB and the
Parquet lives on the data volume, hence download links survive restarts.
"""
from __future__ import annotations

import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import db

JOB_ID_RE = re.compile(r"^h_\d{8}T\d{6}Z_[0-9a-f]{8}$")

_queue: "queue.Queue[str]" = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()

# Test-only job-specific faults: job_id -> exception. Always empty in
# production. Keyed by job id so cross-test leftovers on the shared worker
# thread can never consume a fault meant for another job. The fault is
# raised after aggregation, exercising the same failure/cleanup path.
_failure_hooks: dict[str, Exception] = {}

# Test-only gate invoked with the job id immediately after the worker picks
# a job up, before any processing. Always None in production.
_entry_gate = None


def arm_failure(job_id: str, exc: Exception) -> None:
    """Test-only: make the job ``job_id`` fail with ``exc``."""
    _failure_hooks[job_id] = exc


def drain_queue() -> None:
    """Test-only: drop queued jobs left over from a previous test."""
    while True:
        try:
            _queue.get_nowait()
            _queue.task_done()
        except queue.Empty:
            return


class HandoffError(Exception):
    """A client-facing handoff failure; maps to HTTP 422."""


@dataclass(frozen=True)
class HandoffJob:
    job_id: str
    gen_round1: int
    gen_round2: int
    status: str
    created_at: object
    finished_at: object | None
    file_name: str | None
    n_rows: int | None
    reason: str | None

    @classmethod
    def from_row(cls, row: tuple) -> "HandoffJob":
        return cls(**dict(zip(db.HANDOFF_COLUMNS, row)))

    def to_payload(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "generation_round1": self.gen_round1,
            "generation_round2": self.gen_round2,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "file_name": self.file_name,
            "n_rows": self.n_rows,
            "reason": self.reason,
            "download_url": (
                f"/api/handoff/jobs/{self.job_id}/download" if self.status == "ready" else None
            ),
        }


def requeue_running_jobs() -> int:
    """Re-enqueue jobs left in "running" (e.g. after a service restart).

    Returns the number of jobs requeued. Safe to call with the worker
    already running: it simply picks the requeued ids up.
    """
    rows = db.list_running_handoff_jobs()
    for row in rows:
        _queue.put(row[0])
    return len(rows)


def start_worker() -> None:
    """Start the single background worker (idempotent).

    Jobs left "running" by a previous process are requeued only when the
    worker thread is actually (re)started: callers may invoke this on every
    job creation without enqueueing the same job twice.
    """
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        requeue_running_jobs()
        thread = threading.Thread(target=_worker_loop, name="handoff-worker", daemon=True)
        thread.start()
        _worker_started = True


def create_job() -> HandoffJob:
    """Pin both current generations and enqueue a handover job."""
    gen1 = db.acquire_snapshot(1)
    gen2 = db.acquire_snapshot(2)
    if gen1 is None or gen2 is None:
        missing = [r for r, g in ((1, gen1), (2, gen2)) if g is None]
        labels = "、".join(f"第{'一二'[r - 1]}轮" for r in missing)
        for gen in (gen1, gen2):
            if gen is not None:
                db.release_snapshot(gen)
                db.drop_if_unreferenced(gen)
        raise HandoffError(f"缺少第{labels}数据，两轮齐备后才能生成交接文件")

    job_id = "h_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + os.urandom(4).hex()
    try:
        db.insert_handoff_job(job_id, gen1, gen2)
    except Exception:
        _release_both(gen1, gen2)
        raise
    start_worker()
    _queue.put(job_id)
    return get_job(job_id)  # type: ignore[return-value]


def get_job(job_id: str) -> HandoffJob | None:
    if not JOB_ID_RE.match(job_id or ""):
        return None
    row = db.get_handoff_job(job_id)
    return HandoffJob.from_row(row) if row else None


def list_jobs(limit: int = 50) -> list[HandoffJob]:
    return [HandoffJob.from_row(row) for row in db.list_handoff_jobs(limit)]


def job_file(job: HandoffJob) -> Path:
    return db.handoff_dir() / job.file_name


# ---------------------------------------------------------------- worker
def _worker_loop() -> None:
    while True:
        job_id = _queue.get()
        try:
            if _entry_gate is not None:
                _entry_gate(job_id)
            _process_job(job_id)
        except Exception as exc:  # pragma: no cover - last-resort guard
            try:
                db.mark_handoff_failed(job_id, f"内部错误: {exc}"[:1000])
            except Exception:
                pass
        finally:
            _queue.task_done()


def _process_job(job_id: str) -> None:
    job = get_job(job_id)
    if job is None or job.status != "running":
        return
    gen1, gen2 = job.gen_round1, job.gen_round2
    con = db.read_connection()
    tmp_path: Path | None = None
    try:
        if db.generation_info(gen1) is None or db.generation_info(gen2) is None:
            db.mark_handoff_failed(
                job_id, "生成所依据的代际已不存在（可能被清理），请重新发起生成"
            )
            return

        out_dir = db.handoff_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"handoff_{job_id}_g{gen1}_g{gen2}.parquet"
        tmp_path = out_dir / (file_name + f".tmp.{os.getpid()}.{os.urandom(4).hex()}")

        con.execute(
            "CREATE TEMP TABLE handoff_rows ("
            "round INTEGER, generation BIGINT, district VARCHAR, weight_kind VARCHAR, "
            "rep_id VARCHAR, weight_sum DOUBLE, bus_weight_sum DOUBLE)"
        )
        expected_rows = 0
        for round_no, gen in ((1, gen1), (2, gen2)):
            inserted = _append_round(con, round_no, gen)
            expected_rows += inserted

        # Test fault point: after aggregation, before writing the file, so
        # the failure/cleanup path (no download, snapshots released) is real.
        if job_id in _failure_hooks:
            raise _failure_hooks.pop(job_id)

        # Stream the aggregated (small) result set straight to Parquet;
        # DuckDB spills the heavy join/aggregation to its temp directory.
        con.execute(
            f"COPY (SELECT * FROM handoff_rows ORDER BY round, district, "
            "CASE weight_kind WHEN 'main' THEN 0 ELSE 1 END, rep_id) "
            "TO ? (FORMAT PARQUET, COMPRESSION SNAPPY)",
            [str(tmp_path)],
        )
        con.execute("DROP TABLE handoff_rows")

        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise RuntimeError("Parquet 文件未写出或为空")
        actual_rows = con.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(tmp_path)]
        ).fetchone()[0]
        if int(actual_rows) != expected_rows:
            raise RuntimeError(
                f"交接文件行数不完整：期望 {expected_rows} 行，实际 {actual_rows} 行"
            )

        final_path = out_dir / file_name
        os.replace(tmp_path, final_path)
        tmp_path = None
        db.mark_handoff_ready(job_id, file_name, expected_rows)
    except Exception as exc:
        db.mark_handoff_failed(job_id, f"生成失败: {exc}"[:1000])
    finally:
        try:
            con.close()
        except Exception:
            pass
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        _release_both(gen1, gen2)


def _append_round(con, round_no: int, gen: int) -> int:
    """Append main + replicate aggregate rows of one round; return row count."""
    s = db.samples_table(gen)
    r = db.replicate_table(gen)
    con.execute(
        f"""
        INSERT INTO handoff_rows
        WITH main_rows AS (
            SELECT CAST(? AS INTEGER)               AS round,
                   CAST(? AS BIGINT)                AS generation,
                   s.district                       AS district,
                   'main'                           AS weight_kind,
                   CAST(NULL AS VARCHAR)            AS rep_id,
                   COALESCE(sum(s.main_weight), 0.0)       AS weight_sum,
                   COALESCE(sum(s.main_weight * s.bus), 0.0) AS bus_weight_sum
            FROM "{s}" s
            GROUP BY s.district
        ),
        rep_rows AS (
            SELECT CAST(? AS INTEGER)               AS round,
                   CAST(? AS BIGINT)                AS generation,
                   s.district                       AS district,
                   'replicate'                      AS weight_kind,
                   r.rep_id                         AS rep_id,
                   COALESCE(sum(r.weight), 0.0)       AS weight_sum,
                   COALESCE(sum(r.weight * s.bus), 0.0) AS bus_weight_sum
            FROM "{r}" r
            JOIN "{s}" s ON s.sample_id = r.sample_id
            GROUP BY s.district, r.rep_id
        )
        SELECT * FROM main_rows
        UNION ALL
        SELECT * FROM rep_rows
        """,
        [round_no, gen, round_no, gen],
    )
    return int(
        con.execute(
            "SELECT count(*) FROM handoff_rows WHERE round = ?", [round_no]
        ).fetchone()[0]
    )


def _release_both(gen1: int, gen2: int) -> None:
    for gen in (gen1, gen2):
        try:
            db.release_snapshot(gen)
            db.drop_if_unreferenced(gen)
        except Exception:
            pass
