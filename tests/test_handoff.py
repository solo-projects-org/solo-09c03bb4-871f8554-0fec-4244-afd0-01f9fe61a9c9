"""Tests for the handover Parquet generation feature."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import duckdb
import pytest

from app import db, handoff
from tests.factories import upload, write_csv, write_parquet


def _wait_worker_idle(timeout=10.0) -> None:
    """Block until the shared daemon worker has drained its queue."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if handoff._queue.empty():
            # The worker may have just popped the last item; give it a beat.
            time.sleep(0.05)
            if handoff._queue.empty():
                return
        time.sleep(0.02)
    raise AssertionError("handoff worker did not become idle")


def _pause_worker():
    """Block the worker right after it picks up its next job.

    Returns a ``(job_holder, release)`` pair: ``job_holder`` is a one-element
    list populated with the paused job id, ``release`` resumes processing.
    """
    entered = threading.Event()
    proceed = threading.Event()
    job_holder: list[str] = []

    def gate(job_id: str) -> None:
        job_holder.append(job_id)
        entered.set()
        proceed.wait(10)

    handoff._entry_gate = gate

    def release() -> None:
        if not entered.wait(5):
            handoff._entry_gate = None
            raise AssertionError("worker never entered the gate")
        handoff._entry_gate = None
        proceed.set()

    return job_holder, release


@pytest.fixture(autouse=True)
def _isolated_worker(app_env):
    # The worker thread outlives individual tests. Setup runs only after the
    # shared worker is provably idle, so leftovers from a previous test have
    # finished and resetting test hooks cannot race an in-flight job.
    # app_env is requested first so the worker exists and the DB is initialised.
    _wait_worker_idle()
    handoff.drain_queue()
    handoff._failure_hooks.clear()
    handoff._entry_gate = None
    yield
    # Do not clear state here: wait for this test's own jobs to finish first.
    _wait_worker_idle()


def _wait_status(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/api/handoff/jobs/{job_id}").json()
        if last["status"] != "running":
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still running: {last}")


def _download_parquet(client, job_id, tmp_path: Path) -> Path:
    resp = client.get(f"/api/handoff/jobs/{job_id}/download")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/vnd.apache.parquet"
    out = tmp_path / "downloaded.parquet"
    out.write_bytes(resp.content)
    return out


def _rows(path: Path):
    con = duckdb.connect()
    try:
        return con.execute(
            "SELECT * FROM read_parquet(?) "
            "ORDER BY round, district, CASE weight_kind WHEN 'main' THEN 0 ELSE 1 END, rep_id",
            [str(path)],
        ).fetchall()
    finally:
        con.close()


def test_create_requires_both_rounds(app_env, paths):
    client, _ = app_env
    r1_csv, r1_pq, _, _ = paths
    upload(client, 1, r1_csv, r1_pq)
    resp = client.post("/api/handoff/jobs")
    assert resp.status_code == 422
    assert "第二轮" in resp.json()["detail"]
    assert client.get("/api/handoff/jobs").json()["jobs"] == []


def test_handoff_happy_path_content(app_env, paths, tmp_path):
    client, data_dir = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)

    resp = client.post("/api/handoff/jobs")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert resp.json()["status"] == "running"
    assert handoff.JOB_ID_RE.match(job_id)

    final = _wait_status(client, job_id)
    assert final["status"] == "ready"
    assert final["reason"] is None and final["n_rows"] == 12
    assert final["generation_round1"] == 1 and final["generation_round2"] == 2
    # A finished job exposes its stable download link.
    assert final["download_url"] == f"/api/handoff/jobs/{job_id}/download"

    path = _download_parquet(client, job_id, tmp_path)
    rows = _rows(path)
    # Expected layout: 2 rounds x 2 districts x (1 main + 2 replicates) = 12.
    assert len(rows) == 12
    # Spot-check hand-computed aggregates.
    assert (1, 1, "A", "main", None, 10.0, 4.0) in rows
    assert (1, 1, "A", "replicate", "rep1", 10.0, 8.0) in rows
    assert (1, 1, "A", "replicate", "rep2", 10.0, 0.0) in rows
    assert (2, 2, "A", "main", None, 10.0, 10.0) in rows
    assert (2, 2, "B", "replicate", "rep1", 5.0, 0.0) in rows
    # The file physically lives on the data volume.
    assert (data_dir / "handoff").is_dir()
    assert any((data_dir / "handoff").iterdir())


def test_mismatched_replicate_ids_delivered_as_is(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, r1_pq, r2_csv, _ = paths
    upload(client, 1, r1_csv, r1_pq)
    r2_mismatch_pq = tmp_path / "r2_mismatch.parquet"
    write_parquet(r2_mismatch_pq, [
        ("t1", "rep1", 5.0), ("t2", "rep1", 5.0), ("t3", "rep1", 5.0),
        ("t1", "rep3", 6.0), ("t2", "rep3", 4.0), ("t3", "rep3", 5.0),
    ])
    upload(client, 2, r2_csv, r2_mismatch_pq)

    job_id = client.post("/api/handoff/jobs").json()["job_id"]
    final = _wait_status(client, job_id)
    assert final["status"] == "ready"
    path = _download_parquet(client, job_id, tmp_path)
    rows = _rows(path)
    r1_reps = {row[4] for row in rows if row[0] == 1 and row[3] == "replicate"}
    r2_reps = {row[4] for row in rows if row[0] == 2 and row[3] == "replicate"}
    assert r1_reps == {"rep1", "rep2"}
    assert r2_reps == {"rep1", "rep3"}
    # Row counts stay independent: each round keeps its own id set.
    assert sum(1 for row in rows if row[0] == 1) == 6
    assert sum(1 for row in rows if row[0] == 2) == 6


def test_zero_weight_rows_are_kept(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    # Round 2 with an extra district-C sample whose replicate weights are all
    # zero: replicate rows for C must still appear, with zero sums.
    r2_csv2 = tmp_path / "r2_extra.csv"
    write_csv(r2_csv2, [
        ("t1", "A", 1, 9.0), ("t2", "A", 1, 1.0),
        ("t3", "B", 0, 5.0), ("t4", "C", 1, 3.0),
    ])
    r2_pq2 = tmp_path / "r2_extra.parquet"
    write_parquet(r2_pq2, [
        ("t1", "rep1", 5.0), ("t2", "rep1", 5.0), ("t3", "rep1", 5.0), ("t4", "rep1", 0.0),
        ("t1", "rep2", 6.0), ("t2", "rep2", 4.0), ("t3", "rep2", 5.0), ("t4", "rep2", 0.0),
    ])
    upload(client, 2, r2_csv2, r2_pq2)

    job_id = client.post("/api/handoff/jobs").json()["job_id"]
    final = _wait_status(client, job_id)
    assert final["status"] == "ready"
    path = _download_parquet(client, job_id, tmp_path)
    rows = _rows(path)
    zero_rows = [row for row in rows if row[2] == "C"]
    assert (2, 2, "C", "main", None, 3.0, 3.0) in zero_rows
    assert (2, 2, "C", "replicate", "rep1", 0.0, 0.0) in zero_rows
    assert (2, 2, "C", "replicate", "rep2", 0.0, 0.0) in zero_rows


def test_swap_during_generation_keeps_original_rounds(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)

    job_holder, release = _pause_worker()
    try:
        job_id = client.post("/api/handoff/jobs").json()["job_id"]
        assert job_holder and job_holder[0] == job_id
        # Replace round 2 while the worker holds snapshots of generations 1/2.
        new_csv = tmp_path / "r2_new.csv"
        write_csv(new_csv, [("u1", "A", 0, 5.0), ("u2", "A", 1, 5.0)])
        new_pq = tmp_path / "r2_new.parquet"
        write_parquet(new_pq, [
            ("u1", "rep1", 3.0), ("u2", "rep1", 7.0),
            ("u1", "rep2", 9.0), ("u2", "rep2", 1.0),
        ])
        swap = upload(client, 2, new_csv, new_pq)
        assert swap.status_code == 200
        new_gen = swap.json()["generation"]
    finally:
        release()

    final = _wait_status(client, job_id)
    assert final["status"] == "ready"
    # The job still points at the generations captured at creation time.
    assert final["generation_round2"] == 2
    assert new_gen > 2
    path = _download_parquet(client, job_id, tmp_path)
    rows = _rows(path)
    # Old round-2 data (t-samples, district B) is what the file contains.
    assert (2, 2, "B", "main", None, 5.0, 0.0) in rows
    assert not any(row[2] == "B" and row[0] == 2 and row[1] == new_gen for row in rows)
    # A subsequent job picks up the newly published generations.
    job2 = client.post("/api/handoff/jobs").json()
    final2 = _wait_status(client, job2["job_id"])
    assert final2["generation_round2"] == new_gen


def test_restart_keeps_finished_job_and_requeues_running(app_env, paths, tmp_path):
    client, data_dir = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)

    job_id = client.post("/api/handoff/jobs").json()["job_id"]
    final = _wait_status(client, job_id)
    assert final["status"] == "ready"
    before = client.get(f"/api/handoff/jobs/{job_id}/download").content

    # Simulate a service restart against the same data volume: the on-disk
    # metadata and Parquet survive; the in-memory queue is empty afterwards.
    db._writer.close()
    db._writer = None
    handoff.drain_queue()

    # A job stranded in "running" by the dead process, inserted directly
    # into the metadata table as a real restart would find it.
    db.init_db()
    gen1, gen2 = db.current_generation(1), db.current_generation(2)
    stranded = "h_20000101T000000Z_deadbeef"
    db.insert_handoff_job(stranded, gen1, gen2)

    # Lifespan re-enqueues leftover running jobs; the (already running)
    # worker then completes them against the reopened database.
    assert handoff.requeue_running_jobs() == 1
    final2 = _wait_status(client, stranded)
    assert final2["status"] == "ready"
    assert final2["n_rows"] == 12

    # The previously finished job's link still serves the exact same file,
    # proving download links survive a restart.
    again = client.get(f"/api/handoff/jobs/{job_id}/download")
    assert again.status_code == 200
    assert again.content == before


def test_failed_job_reports_reason_and_blocks_download(app_env, paths):
    client, _ = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)

    # Pause the shared worker at the entry gate so the fault is armed
    # before any processing of this job starts.
    job_holder, release = _pause_worker()
    try:
        job_id = client.post("/api/handoff/jobs").json()["job_id"]
        assert job_holder and job_holder[0] == job_id
        handoff.arm_failure(job_id, RuntimeError("模拟聚合失败"))
    finally:
        release()
    final = _wait_status(client, job_id)
    assert final["status"] == "failed"
    assert "模拟聚合失败" in final["reason"]
    assert final["download_url"] is None
    resp = client.get(f"/api/handoff/jobs/{job_id}/download")
    assert resp.status_code == 409
    assert "模拟聚合失败" in resp.json()["detail"]

    # The failed job's snapshots were released; a fresh job succeeds.
    job2 = client.post("/api/handoff/jobs").json()
    assert _wait_status(client, job2["job_id"])["status"] == "ready"


def test_unknown_job_and_bad_id(app_env):
    client, _ = app_env
    assert client.get("/api/handoff/jobs/h_does_not_exist").status_code == 404
    assert client.get("/api/handoff/jobs/../etc").status_code == 404
    # Well-formed but unknown id.
    assert client.get("/api/handoff/jobs/h_20000101T000000Z_00000000").status_code == 404


def test_jobs_listed_newest_first(app_env, paths):
    client, _ = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)
    ids = []
    for _ in range(3):
        ids.append(client.post("/api/handoff/jobs").json()["job_id"])
        time.sleep(0.01)
    for job_id in ids:
        _wait_status(client, job_id)
    listed = [j["job_id"] for j in client.get("/api/handoff/jobs").json()["jobs"]]
    assert listed == list(reversed(ids))
