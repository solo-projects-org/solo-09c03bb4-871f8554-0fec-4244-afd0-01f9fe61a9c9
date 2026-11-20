"""Quantile implementation cross-check and extra validation tests."""
from __future__ import annotations

import random

import duckdb
import pytest

from app.analysis import _quantiles
from .factories import upload, write_parquet


def test_quantiles_match_duckdb_quantile_cont():
    rng = random.Random(20261120)
    con = duckdb.connect()
    try:
        con.execute("CREATE TEMP TABLE q (x DOUBLE)")
        for n in (1, 2, 3, 7, 50, 501):
            values = [rng.uniform(-1, 1) for _ in range(n)]
            low, high = _quantiles(values)
            con.execute("DELETE FROM q")
            con.executemany("INSERT INTO q VALUES (?)", [(v,) for v in values])
            ref = con.execute(
                "SELECT quantile_cont(x, [0.025, 0.975]) FROM q"
            ).fetchone()[0]
            assert low == pytest.approx(float(ref[0]), rel=1e-12, abs=1e-12)
            assert high == pytest.approx(float(ref[1]), rel=1e-12, abs=1e-12)
    finally:
        con.close()


def test_zero_sum_replicate_rejected(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, _, _, _ = paths
    # rep2 is all zeros: every (sample, rep2) row exists but the total is 0.
    bad_pq = tmp_path / "zero_sum.parquet"
    write_parquet(bad_pq, [
        ("s1", "rep1", 8.0), ("s2", "rep1", 2.0), ("s3", "rep1", 2.0),
        ("s1", "rep2", 0.0), ("s2", "rep2", 0.0), ("s3", "rep2", 0.0),
    ])
    r = upload(client, 1, r1_csv, bad_pq)
    assert r.status_code == 422
    assert "总权重为零" in r.json()["detail"]


def test_inf_weight_rejected(app_env, paths, tmp_path):
    # Infinity cannot be represented through the parquet writer portably;
    # build the parquet with DuckDB directly.
    client, _ = app_env
    r1_csv, _, _, _ = paths
    bad_pq = tmp_path / "inf.parquet"
    con = duckdb.connect()
    try:
        con.execute(
            """
            CREATE TEMP TABLE r AS
            SELECT unnest(['s1','s2','s3','s1','s2','s3']) AS sample_id,
                   unnest(['rep1','rep1','rep1','rep2','rep2','rep2']) AS rep_id,
                   unnest([8.0, 2.0, 2.0, 'inf'::DOUBLE, 10.0, 2.0]) AS weight
            """
        )
        con.execute("COPY r TO ? (FORMAT PARQUET)", [str(bad_pq)])
    finally:
        con.close()
    r = upload(client, 1, r1_csv, bad_pq)
    assert r.status_code == 422 and "非负" in r.json()["detail"]
