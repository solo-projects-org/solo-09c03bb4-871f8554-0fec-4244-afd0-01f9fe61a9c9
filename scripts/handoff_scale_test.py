"""End-to-end scale test for handover Parquet generation.

Spawns the real uvicorn server (DuckDB memory_limit=180MB, like the Docker
image), uploads two rounds with at least 5,000,000 replicate-weight rows
each, then:

1. creates a handover job, waits for it and downloads the Parquet,
2. independently recomputes every aggregate from the source files and
   compares them row by row against the delivered file,
3. starts a second handover job and swaps round 2 while it is running,
   proving the delivered file still matches the generations captured when
   the job was created,
4. records the service process peak RSS along the way.

Run:
  python scripts/handoff_scale_test.py --rows 5000000 --out /tmp/handoff_scale
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from tests.factories import make_large  # noqa: E402


def post_files(url: str, csv_path: Path, pq_path: Path) -> tuple[int, dict]:
    boundary = "----handoffboundary" + os.urandom(8).hex()
    parts = []
    for field, path in (("sample_csv", csv_path), ("replicate_parquet", pq_path)):
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n\r\n'.encode()
        )
        parts.append(path.read_bytes())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=900) as resp:
        return json.loads(resp.read().decode())


def post_json(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def wait_ready(base: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            get_json(base + "/api/status")
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not become ready")


def wait_job(base: str, job_id: str, timeout: float = 900.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = get_json(f"{base}/api/handoff/jobs/{job_id}")
        if last["status"] != "running":
            return last
        time.sleep(0.5)
    raise RuntimeError(f"job {job_id} stuck: {last}")


def rss_mib(pid: int) -> float:
    statm = Path(f"/proc/{pid}/statm").read_text().split()
    page_kb = resource.getpagesize() // 1024
    return int(statm[1]) * page_kb / 1024


def rss_hwm_mib(pid: int) -> float:
    """Historical peak resident set size (VmHWM) since the process started."""
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return rss_mib(pid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5_000_000)
    ap.add_argument("--out", default="/tmp/handoff_scale")
    ap.add_argument("--duckdb-mem", default="180MB")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)

    n_reps = min(500, args.rows)
    n_samples = max(1, args.rows // n_reps)
    print(f"[plan] {n_samples} samples x {n_reps} reps = {n_samples * n_reps:,} rows per round")

    print("[gen] round 1 ...")
    t0 = time.time()
    r1_csv, r1_pq = make_large(out / "r1", n_samples, n_reps, seed=7)
    print(f"      {time.time()-t0:.1f}s, parquet {r1_pq.stat().st_size/1e6:.1f} MiB")
    print("[gen] round 2 ...")
    t0 = time.time()
    r2_csv, r2_pq = make_large(out / "r2", n_samples, n_reps, seed=23)
    print(f"      {time.time()-t0:.1f}s, parquet {r2_pq.stat().st_size/1e6:.1f} MiB")

    env = os.environ.copy()
    env.update({
        "DATA_DIR": str(data_dir),
        "UPLOAD_TMP_DIR": str(data_dir / "tmp"),
        "DUCKDB_MEMORY_LIMIT": args.duckdb_mem,
        "DUCKDB_THREADS": "2",
        "PYTHONPATH": str(ROOT),
    })
    cmd = [sys.executable, "-m", "uvicorn",
           "app.main:app", "--host", "127.0.0.1", "--port", "8124", "--workers", "1"]
    log = open(out / "server.log", "wb")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
    base = "http://127.0.0.1:8124"
    try:
        wait_ready(base)
        print(f"[srv] pid={proc.pid} baseline RSS={rss_mib(proc.pid):.1f} MiB")

        for round_no, csv_p, pq_p in ((1, r1_csv, r1_pq), (2, r2_csv, r2_pq)):
            t0 = time.time()
            code, payload = post_files(f"{base}/api/rounds/{round_no}", csv_p, pq_p)
            print(f"[up] round {round_no}: HTTP {code} in {time.time()-t0:.1f}s")
            if code != 200:
                print(payload)
                return 2
        gen1 = get_json(base + "/api/status")["rounds"]["1"]["generation"]
        gen2 = get_json(base + "/api/status")["rounds"]["2"]["generation"]
        print(f"[up] published generations: round1={gen1} round2={gen2}")

        # ---- job 1: plain generation ---------------------------------------
        # Sample RSS in a background thread so the aggregation-time peak is
        # captured rather than only the post-completion value.
        peak = {"rss": 0.0}
        stop_sampler = threading.Event()

        def sample_rss() -> None:
            while not stop_sampler.wait(0.1):
                peak["rss"] = max(peak["rss"], rss_mib(proc.pid))

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        t0 = time.time()
        code, job = post_json(base + "/api/handoff/jobs")
        assert code == 202, job
        job_id = job["job_id"]
        final = wait_job(base, job_id)
        stop_sampler.set()
        sampler.join(timeout=2)
        dt = time.time() - t0
        print(f"[job1] {final['status']} in {dt:.1f}s -> {final['n_rows']} rows, "
              f"file {final['file_name']}")
        assert final["status"] == "ready", final
        assert final["generation_round1"] == gen1 and final["generation_round2"] == gen2
        print(f"[mem] sampled RSS peak during generation: {peak['rss']:.1f} MiB; "
              f"process VmHWM: {rss_hwm_mib(proc.pid):.1f} MiB")

        dl_path = out / final["file_name"]
        with urllib.request.urlopen(base + final["download_url"]) as resp:
            dl_path.write_bytes(resp.read())
        print(f"[dl] {dl_path.name} {dl_path.stat().st_size/1e6:.2f} MiB on disk")

        # ---- independent recomputation from source files ------------------
        ref = duckdb.connect()
        ref.execute(
            """
            CREATE TEMP TABLE expected (
                round INTEGER, generation BIGINT, district VARCHAR,
                weight_kind VARCHAR, rep_id VARCHAR,
                weight_sum DOUBLE, bus_weight_sum DOUBLE)
            """
        )
        ref.execute(
            """
            INSERT INTO expected
            SELECT 1, ?::BIGINT, district, 'main', NULL,
                   sum(main_weight), sum(main_weight*bus)
            FROM read_csv_auto(?) GROUP BY district
            UNION ALL
            SELECT 1, ?::BIGINT, s.district, 'replicate', r.rep_id,
                   sum(r.weight), sum(r.weight*s.bus)
            FROM read_parquet(?) r JOIN read_csv_auto(?) s USING (sample_id)
            GROUP BY s.district, r.rep_id
            UNION ALL
            SELECT 2, ?::BIGINT, district, 'main', NULL,
                   sum(main_weight), sum(main_weight*bus)
            FROM read_csv_auto(?) GROUP BY district
            UNION ALL
            SELECT 2, ?::BIGINT, s.district, 'replicate', r.rep_id,
                   sum(r.weight), sum(r.weight*s.bus)
            FROM read_parquet(?) r JOIN read_csv_auto(?) s USING (sample_id)
            GROUP BY s.district, r.rep_id
            """,
            [gen1, str(r1_csv),
             gen1, str(r1_pq), str(r1_csv),
             gen2, str(r2_csv),
             gen2, str(r2_pq), str(r2_csv)],
        )
        n_expected = ref.execute("SELECT count(*) FROM expected").fetchone()[0]
        n_actual = ref.execute("SELECT count(*) FROM read_parquet(?)", [str(dl_path)]).fetchone()[0]
        print(f"[xref] expected rows={n_expected}, delivered rows={n_actual}")
        assert n_expected == n_actual == final["n_rows"]

        # Parallel DuckDB aggregation sums in different orders, so bit-exact
        # equality cannot be expected across two independent runs; compare
        # with a tight relative tolerance (machine-epsilon scale).
        bad = ref.execute(
            """
            SELECT count(*) FROM expected e
            JOIN read_parquet(?) d
              ON e.round = d.round AND e.district = d.district
             AND e.weight_kind = d.weight_kind
             AND COALESCE(e.rep_id, '') = COALESCE(d.rep_id, '')
            WHERE abs(e.weight_sum - d.weight_sum)
                    > 1e-9 * greatest(abs(e.weight_sum), 1.0)
               OR abs(e.bus_weight_sum - d.bus_weight_sum)
                    > 1e-9 * greatest(abs(e.bus_weight_sum), 1.0)
            """,
            [str(dl_path)],
        ).fetchone()[0]
        assert bad == 0, f"{bad} rows exceed the float tolerance"
        max_diff = ref.execute(
            """
            SELECT max(greatest(
                abs(e.weight_sum - d.weight_sum),
                abs(e.bus_weight_sum - d.bus_weight_sum)))
            FROM expected e JOIN read_parquet(?) d
              ON e.round = d.round AND e.district = d.district
             AND e.weight_kind = d.weight_kind
             AND COALESCE(e.rep_id, '') = COALESCE(d.rep_id, '')
            """,
            [str(dl_path)],
        ).fetchone()[0]
        print(f"[xref] all rows match within 1e-9 relative tolerance "
              f"(max abs diff {max_diff:.3e})")
        ref.close()

        # ---- job 2: swap round 2 while the job is being built -------------
        code, job2 = post_json(base + "/api/handoff/jobs")
        assert code == 202, job2
        # Give the worker a moment to pin its snapshot, then swap round 2.
        time.sleep(1.0)
        t0 = time.time()
        code_swap, swap = post_files(f"{base}/api/rounds/2", r2_csv, r2_pq)
        print(f"[swap] HTTP {code_swap} in {time.time()-t0:.1f}s during job {job2['job_id']}")
        assert code_swap == 200
        new_gen2 = swap["generation"]
        assert new_gen2 != gen2
        final2 = wait_job(base, job2["job_id"])
        print(f"[job2] {final2['status']} gens=({final2['generation_round1']},"
              f"{final2['generation_round2']}) rows={final2['n_rows']}")
        assert final2["status"] == "ready"
        # The in-flight job kept the original round-2 generation.
        assert final2["generation_round2"] == gen2
        dl2 = out / final2["file_name"]
        with urllib.request.urlopen(base + final2["download_url"]) as resp:
            dl2.write_bytes(resp.read())
        chk = duckdb.connect()
        gens = chk.execute(
            "SELECT DISTINCT round, generation FROM read_parquet(?) ORDER BY 1",
            [str(dl2)],
        ).fetchall()
        chk.close()
        assert gens == [(1, gen1), (2, gen2)], gens
        print("[swap] delivered file still corresponds to the original two rounds")

        # A job created after the swap sees the new generation.
        code, job3 = post_json(base + "/api/handoff/jobs")
        final3 = wait_job(base, job3["job_id"])
        assert final3["generation_round2"] == new_gen2
        print(f"[swap] later job correctly uses new generation {new_gen2}")

        print(f"[mem] final process VmHWM: {rss_hwm_mib(proc.pid):.1f} MiB "
              f"(service cap 384 MiB)")
        print("[OK] all handoff scale checks passed")
        return 0
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
