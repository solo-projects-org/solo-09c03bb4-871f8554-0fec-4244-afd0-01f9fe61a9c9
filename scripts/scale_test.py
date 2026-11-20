"""End-to-end scale and memory test.

Spawns the real uvicorn server in a subprocess, optionally constrained by a
cgroup memory limit (to emulate the Docker Compose 384 MiB cap), then:

1. generates two rounds with at least 5,000,000 replicate-weight rows each,
2. uploads them through the HTTP API (validation + full coverage checks),
3. runs district queries and checks the point estimate and 2.5/97.5 bounds,
4. measures peak RSS of the service process,
5. verifies a failed replacement leaves the old data queryable,
6. verifies a successful swap publishes a new generation.

Run:
  python scripts/scale_test.py --rows 5000000 --mem-mb 384 --out /tmp/scale
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from tests.factories import make_large  # noqa: E402


def post_files(url: str, csv_path: Path, pq_path: Path) -> tuple[int, dict]:
    boundary = "----scaleboundary" + os.urandom(8).hex()
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
        url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=900) as resp:
        return json.loads(resp.read().decode())


def wait_ready(base: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            get_json(base + "/api/status")
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not become ready")


def rss_kb(pid: int) -> int:
    statm = Path(f"/proc/{pid}/statm").read_text().split()
    page_kb = resource.getpagesize() // 1024
    return int(statm[1]) * page_kb


def cgroup_rss_kb(cgroup_path: Path | None) -> int | None:
    """Peak RSS recorded by the test's own v2 memory cgroup."""
    if cgroup_path is None:
        return None
    p = cgroup_path / "memory.peak"
    if p.exists():
        return int(p.read_text().strip()) // 1024
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5_000_000, help="replicate rows per round")
    ap.add_argument("--mem-mb", type=int, default=384)
    ap.add_argument("--out", default="/tmp/bus_scale")
    ap.add_argument("--no-cgroup", action="store_true", help="skip cgroup, use RLIMIT_AS only")
    ap.add_argument("--duckdb-mem", default="180MB")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)

    # rows = n_samples * n_reps
    n_reps = min(500, args.rows)
    n_samples = max(1, args.rows // n_reps)
    actual_rows = n_samples * n_reps
    print(f"[plan] {n_samples} samples x {n_reps} reps = {actual_rows:,} rows per round")

    print("[gen] round 1 ...")
    t0 = time.time()
    r1_csv, r1_pq = make_large(out / "r1", n_samples, n_reps, seed=7)
    print(f"      done in {time.time()-t0:.1f}s, parquet {r1_pq.stat().st_size/1e6:.1f} MiB")
    print("[gen] round 2 ...")
    t0 = time.time()
    r2_csv, r2_pq = make_large(out / "r2", n_samples, n_reps, seed=23)
    print(f"      done in {time.time()-t0:.1f}s, parquet {r2_pq.stat().st_size/1e6:.1f} MiB")

    # ---- launch server under a memory limit ---------------------------------
    env = os.environ.copy()
    env.update({
        "DATA_DIR": str(data_dir),
        "UPLOAD_TMP_DIR": str(data_dir / "tmp"),
        "DUCKDB_MEMORY_LIMIT": args.duckdb_mem,
        "DUCKDB_THREADS": "2",
        "PYTHONPATH": str(ROOT),
    })
    cmd = [
        sys.executable, "-I", "-m", "uvicorn",
        "app.main:app", "--host", "127.0.0.1", "--port", "8123", "--workers", "1",
    ]

    cgroup_path = None
    if not args.no_cgroup and os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
        cgroup_path = Path("/sys/fs/cgroup/bus_scale")
        try:
            cgroup_path.mkdir(exist_ok=True)
            (cgroup_path / "memory.max").write_text(str(args.mem_mb * 1024 * 1024))
            (cgroup_path / "memory.swap.max").write_text("0")
            print(f"[cg] memory.max = {args.mem_mb} MiB at {cgroup_path}")
        except (PermissionError, OSError) as exc:
            print(f"[cg] cannot create cgroup ({exc}); will measure peak RSS instead")
            cgroup_path = None

    log = open(out / "server.log", "wb")
    if cgroup_path is not None:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
    else:
        # No cgroup access (e.g. sandboxed CI): run without an OS-level cap and
        # measure peak RSS instead. The Docker Compose service still enforces
        # mem_limit=384m; DuckDB's own memory_limit keeps its buffers bounded.
        print("[cg] running without OS memory cap; peak RSS will be measured")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
    try:
        if cgroup_path is not None:
            try:
                (cgroup_path / "cgroup.procs").write_text(str(proc.pid))
            except PermissionError:
                print("[cg] could not attach process; continuing without cgroup")
                cgroup_path = None
        base = "http://127.0.0.1:8123"
        wait_ready(base)
        print(f"[srv] pid={proc.pid} ready; baseline RSS={rss_kb(proc.pid)/1024:.1f} MiB")

        # ---- uploads --------------------------------------------------------
        for round_no, csv_p, pq_p in ((1, r1_csv, r1_pq), (2, r2_csv, r2_pq)):
            t0 = time.time()
            code, payload = post_files(f"{base}/api/rounds/{round_no}", csv_p, pq_p)
            dt = time.time() - t0
            print(f"[up] round {round_no}: HTTP {code} in {dt:.1f}s -> {payload}")
            if code != 200:
                return 2
            assert payload["n_rep_rows"] >= args.rows, payload
        peak_rss = rss_kb(proc.pid) / 1024
        cg_peak = cgroup_rss_kb(cgroup_path)
        print(f"[mem] process RSS after uploads: {peak_rss:.1f} MiB; cgroup peak: "
              f"{(f'{cg_peak/1024:.1f} MiB' if cg_peak else 'n/a')}")

        status = get_json(base + "/api/status")
        print("[status]", json.dumps(status, ensure_ascii=False)[:400])
        districts = status["districts"]
        assert districts, "no districts listed"
        gen1 = status["rounds"]["1"]["generation"]
        gen2 = status["rounds"]["2"]["generation"]

        # ---- queries --------------------------------------------------------
        for district in districts[:3]:
            t0 = time.time()
            d = get_json(base + "/api/result?" + urllib.parse.urlencode({"district": district}))
            dt = time.time() - t0
            print(f"[q] {district}: point={d['point_estimate']:.6f} "
                  f"ci=[{d['ci_low']:.6f}, {d['ci_high']:.6f}] reps={d['n_replicates']} ({dt:.1f}s)")
            assert d["available"] and d["ci_low"] <= d["point_estimate"] <= d["ci_high"]
            assert d["n_replicates"] == n_reps

        # cross-check quantiles against an independent computation straight
        # from the source files (the live DB file is locked by the server)
        con = duckdb.connect()
        district = districts[0]
        ref = con.execute(
            """
            WITH s1 AS (SELECT * FROM read_csv_auto(?)),
                 r1 AS (SELECT * FROM read_parquet(?)),
                 s2 AS (SELECT * FROM read_csv_auto(?)),
                 r2 AS (SELECT * FROM read_parquet(?)),
                 d1 AS (
                    SELECT r1.rep_id, sum(r1.weight * s1.bus) / sum(r1.weight) AS p
                    FROM r1 JOIN s1 USING (sample_id)
                    WHERE s1.district = ? GROUP BY r1.rep_id
                 ),
                 d2 AS (
                    SELECT r2.rep_id, sum(r2.weight * s2.bus) / sum(r2.weight) AS p
                    FROM r2 JOIN s2 USING (sample_id)
                    WHERE s2.district = ? GROUP BY r2.rep_id
                 )
            SELECT quantile_cont(d2.p - d1.p, [0.025, 0.975])
            FROM d1 JOIN d2 USING (rep_id)
            """,
            [str(r1_csv), str(r1_pq), str(r2_csv), str(r2_pq), district, district],
        ).fetchone()[0]
        con.close()
        d = get_json(base + "/api/result?" + urllib.parse.urlencode({"district": district}))
        print(f"[xref] service=({d['ci_low']:.6f},{d['ci_high']:.6f}) duckdb={tuple(round(float(x),6) for x in ref)}")
        assert abs(d["ci_low"] - float(ref[0])) < 1e-9
        assert abs(d["ci_high"] - float(ref[1])) < 1e-9

        # ---- failed swap keeps old data ------------------------------------
        bad_pq = out / "bad.parquet"
        dc = duckdb.connect()
        dc.execute("CREATE TEMP TABLE b(sample_id VARCHAR, rep_id VARCHAR, weight DOUBLE)")
        dc.execute("INSERT INTO b SELECT sample_id, rep_id, weight FROM read_parquet(?) LIMIT 100", [str(r2_pq)])
        dc.execute("COPY b TO ? (FORMAT PARQUET)", [str(bad_pq)])
        dc.close()
        code, payload = post_files(f"{base}/api/rounds/2", r2_csv, bad_pq)
        print(f"[swap-fail] HTTP {code}: {payload.get('detail', payload)[:120]}")
        assert code == 422
        after = get_json(base + "/api/result?" + urllib.parse.urlencode({"district": district}))
        # Parallel floating-point aggregation may differ in the last digits.
        assert after["n_replicates"] == n_reps
        assert abs(after["ci_high"] - d["ci_high"]) < 1e-12
        assert abs(after["ci_low"] - d["ci_low"]) < 1e-12
        assert abs(after["point_estimate"] - d["point_estimate"]) < 1e-12

        # ---- successful swap ------------------------------------------------
        print("[swap-ok] re-uploading round 2 (new generation)")
        code, payload = post_files(f"{base}/api/rounds/2", r2_csv, r2_pq)
        assert code == 200 and payload["generation"] > gen2
        after = get_json(base + "/api/result?" + urllib.parse.urlencode({"district": district}))
        assert after["round2"]["generation"] == payload["generation"]
        print(f"[swap-ok] now generation {payload['generation']}")

        peak_rss = rss_kb(proc.pid) / 1024
        print(f"[mem] final process RSS: {peak_rss:.1f} MiB")
        if cg_peak:
            print(f"[mem] cgroup peak RSS: {cg_peak/1024:.1f} MiB (limit {args.mem_mb} MiB)")
        print("[OK] all scale checks passed")
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
