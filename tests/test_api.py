"""Functional tests: math, validation failures, mismatch, zero denom, swap."""
from __future__ import annotations

import pytest

from .factories import upload, write_csv, write_parquet


def test_status_empty(app_env):
    client, _ = app_env
    s = client.get("/api/status").json()
    assert s["rounds"]["1"]["loaded"] is False
    assert s["rounds"]["2"]["loaded"] is False
    assert s["districts"] == []


def test_happy_path_point_and_ci(app_env, paths):
    client, _ = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    assert upload(client, 1, r1_csv, r1_pq).status_code == 200
    assert upload(client, 2, r2_csv, r2_pq).status_code == 200

    d = client.get("/api/result", params={"district": "A"}).json()
    # main proportions: r1 = 4/10 = 0.4, r2 = 10/10 = 1.0 -> diff 0.6
    assert d["available"] is True
    assert d["reason"] is None
    assert d["round1"]["proportion"] == pytest.approx(0.4)
    assert d["round2"]["proportion"] == pytest.approx(1.0)
    assert d["point_estimate"] == pytest.approx(0.6)
    # replicate proportions in district A:
    #   r1: rep1 = 8/(8+2)=0.8,  rep2 = 0/(0+10)=0
    #   r2: both samples are bus=1, so rep1 = rep2 = 1.0
    # paired diffs: rep1 1.0-0.8=0.2; rep2 1.0-0=1.0
    # type-7 quantiles of [0.2, 1.0] (h = (n-1)*p):
    # q0.025 = 0.22, q0.975 = 0.98
    assert d["ci_low"] == pytest.approx(0.22)
    assert d["ci_high"] == pytest.approx(0.98)
    assert d["n_replicates"] == 2


def test_district_list_and_status(app_env, paths):
    client, _ = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)
    s = client.get("/api/status").json()
    assert s["districts"] == ["A", "B"]
    assert s["rounds"]["1"]["n_samples"] == 3
    assert s["rounds"]["1"]["n_replicates"] == 2
    assert s["rounds"]["1"]["n_rep_rows"] == 6


def test_missing_round_blank(app_env, paths):
    client, _ = app_env
    r1_csv, r1_pq, _, _ = paths
    upload(client, 1, r1_csv, r1_pq)
    d = client.get("/api/result", params={"district": "A"}).json()
    assert d["available"] is False
    assert d["point_estimate"] is None and d["ci_low"] is None
    assert "第二轮" in d["reason"]


def test_replicate_id_set_mismatch(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    # round 2 with rep ids rep1 / rep3 (full coverage, valid on its own)
    r2_bad_pq = tmp_path / "r2_mismatch.parquet"
    write_parquet(r2_bad_pq, [
        ("t1", "rep1", 5.0), ("t2", "rep1", 5.0), ("t3", "rep1", 5.0),
        ("t1", "rep3", 6.0), ("t2", "rep3", 4.0), ("t3", "rep3", 5.0),
    ])
    resp = upload(client, 2, r2_csv, r2_bad_pq)
    assert resp.status_code == 200  # each round validates independently
    d = client.get("/api/result", params={"district": "A"}).json()
    assert d["point_estimate"] == pytest.approx(0.6)  # point still shown
    assert d["ci_low"] is None and d["ci_high"] is None
    assert "重抽编号集合不同" in d["reason"]
    assert "rep3" in d["reason"] and "rep2" in d["reason"]


def test_main_zero_denominator(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)
    # district C exists nowhere -> zero denominators in both rounds
    d = client.get("/api/result", params={"district": "C"}).json()
    assert d["available"] is False
    assert d["point_estimate"] is None
    assert d["ci_low"] is None and d["ci_high"] is None
    assert "分母为零" in d["reason"]


def test_replicate_zero_denominator(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    # Round 2 adds a district-C sample whose replicate weight is zero in
    # every replicate; in district C the replicate denominator is zero.
    r2_extra_csv = tmp_path / "r2_extra.csv"
    write_csv(r2_extra_csv, [
        ("t1", "A", 1, 9.0), ("t2", "A", 1, 1.0),
        ("t3", "B", 0, 5.0), ("t4", "C", 1, 3.0),
    ])
    r2_extra_pq = tmp_path / "r2_extra.parquet"
    write_parquet(r2_extra_pq, [
        ("t1", "rep1", 5.0), ("t2", "rep1", 5.0), ("t3", "rep1", 5.0), ("t4", "rep1", 0.0),
        ("t1", "rep2", 6.0), ("t2", "rep2", 4.0), ("t3", "rep2", 5.0), ("t4", "rep2", 0.0),
    ])
    assert upload(client, 2, r2_extra_csv, r2_extra_pq).status_code == 200
    d = client.get("/api/result", params={"district": "C"}).json()
    # main denominator in round 2 is 3 > 0 but round 1 has no C samples:
    # main zero-denominator case. Build the pure replicate-zero case instead.
    # Add the same sample to round 1 with positive main weight but zero
    # replicate weights via a re-upload of round 1.
    r1_extra_csv = tmp_path / "r1_extra.csv"
    write_csv(r1_extra_csv, [
        ("s1", "A", 1, 4.0), ("s2", "A", 0, 6.0), ("s3", "B", 1, 2.0),
        ("s4", "C", 1, 7.0),
    ])
    r1_extra_pq = tmp_path / "r1_extra.parquet"
    write_parquet(r1_extra_pq, [
        ("s1", "rep1", 8.0), ("s2", "rep1", 2.0), ("s3", "rep1", 2.0), ("s4", "rep1", 0.0),
        ("s1", "rep2", 0.0), ("s2", "rep2", 10.0), ("s3", "rep2", 2.0), ("s4", "rep2", 0.0),
    ])
    assert upload(client, 1, r1_extra_csv, r1_extra_pq).status_code == 200
    d = client.get("/api/result", params={"district": "C"}).json()
    assert d["available"] is True
    assert d["point_estimate"] == pytest.approx((3.0 / 3.0) - (7.0 / 7.0))
    assert d["ci_low"] is None and d["ci_high"] is None
    assert "重抽权重分母为零" in d["reason"]


# ------------------------------------------------------------- validation
def _post_raw(client, round_no, csv_path, pq_path, csv_name="x.csv", pq_name="x.parquet"):
    with open(csv_path, "rb") as f1, open(pq_path, "rb") as f2:
        return client.post(
            f"/api/rounds/{round_no}",
            files={
                "sample_csv": (csv_name, f1.read(), "text/csv"),
                "replicate_parquet": (pq_name, f2.read(), "application/octet-stream"),
            },
        )


def test_bad_extensions(app_env, paths):
    client, _ = app_env
    r1_csv, r1_pq, _, _ = paths
    r = _post_raw(client, 1, r1_csv, r1_pq, csv_name="x.txt")
    assert r.status_code == 422 and ".csv" in r.json()["detail"]
    r = _post_raw(client, 1, r1_csv, r1_pq, pq_name="x.csv")
    assert r.status_code == 422 and ".parquet" in r.json()["detail"]


def test_bad_bus_value(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, r1_pq, _, _ = paths
    bad = tmp_path / "bad_bus.csv"
    write_csv(bad, [("s1", "A", 2, 4.0), ("s2", "A", 0, 6.0), ("s3", "B", 1, 2.0)])
    r = upload(client, 1, bad, r1_pq)
    assert r.status_code == 422 and "0 或 1" in r.json()["detail"]
    assert client.get("/api/status").json()["rounds"]["1"]["loaded"] is False


def test_negative_weight(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, r1_pq, _, _ = paths
    bad_pq = tmp_path / "neg.parquet"
    write_parquet(bad_pq, [
        ("s1", "rep1", -8.0), ("s2", "rep1", 2.0), ("s3", "rep1", 2.0),
        ("s1", "rep2", 0.0), ("s2", "rep2", 10.0), ("s3", "rep2", 2.0),
    ])
    r = upload(client, 1, r1_csv, bad_pq)
    assert r.status_code == 422 and "非负" in r.json()["detail"]


def test_foreign_sample_rejected(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, r1_pq, _, _ = paths
    bad_pq = tmp_path / "foreign.parquet"
    write_parquet(bad_pq, [
        ("s1", "rep1", 8.0), ("s2", "rep1", 2.0), ("ghost", "rep1", 2.0),
        ("s1", "rep2", 0.0), ("s2", "rep2", 10.0), ("s3", "rep2", 2.0),
    ])
    r = upload(client, 1, r1_csv, bad_pq)
    assert r.status_code == 422 and "不存在的样本号" in r.json()["detail"]


def test_missing_replicate_weight_rejected(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, _, _, _ = paths
    bad_pq = tmp_path / "missing.parquet"
    # s3 has no rep1 row -> coverage hole
    write_parquet(bad_pq, [
        ("s1", "rep1", 8.0), ("s2", "rep1", 2.0),
        ("s1", "rep2", 0.0), ("s2", "rep2", 10.0), ("s3", "rep2", 2.0),
    ])
    r = upload(client, 1, r1_csv, bad_pq)
    assert r.status_code == 422 and "恰有一条" in r.json()["detail"]


def test_duplicate_replicate_weight_rejected(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, _, _, _ = paths
    bad_pq = tmp_path / "dup.parquet"
    write_parquet(bad_pq, [
        ("s1", "rep1", 8.0), ("s1", "rep1", 1.0), ("s2", "rep1", 2.0), ("s3", "rep1", 2.0),
        ("s1", "rep2", 0.0), ("s2", "rep2", 10.0), ("s3", "rep2", 2.0),
    ])
    r = upload(client, 1, r1_csv, bad_pq)
    assert r.status_code == 422 and "恰有一条" in r.json()["detail"]


def test_duplicate_sample_id_rejected(app_env, paths, tmp_path):
    client, _ = app_env
    _, r1_pq, _, _ = paths
    bad_csv = tmp_path / "dup_id.csv"
    write_csv(bad_csv, [
        ("s1", "A", 1, 4.0), ("s1", "A", 0, 6.0), ("s3", "B", 1, 2.0),
    ])
    r = upload(client, 1, bad_csv, r1_pq)
    assert r.status_code == 422 and "重复" in r.json()["detail"]


def test_bad_column_names(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, r1_pq, _, _ = paths
    bad_csv = tmp_path / "cols.csv"
    # CSV with an extra column
    bad_csv.write_text("sample_id,district,bus,main_weight,extra\ns1,A,1,4.0,99\n", encoding="utf-8")
    r = upload(client, 1, bad_csv, r1_pq)
    assert r.status_code == 422 and "列结构" in r.json()["detail"]


# ------------------------------------------------------------- swap tests
def test_failed_swap_keeps_old_data(app_env, paths, tmp_path):
    client, _ = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)
    before = client.get("/api/result", params={"district": "A"}).json()

    # Invalid replacement for round 2: rep coverage hole.
    bad_pq = tmp_path / "bad_swap.parquet"
    write_parquet(bad_pq, [
        ("t1", "rep1", 5.0), ("t2", "rep1", 5.0), ("t3", "rep1", 5.0),
        ("t1", "rep2", 6.0),  # t2/t3 missing for rep2
    ])
    r = upload(client, 2, r2_csv, bad_pq)
    assert r.status_code == 422
    after = client.get("/api/result", params={"district": "A"}).json()
    assert after["point_estimate"] == before["point_estimate"]
    assert after["ci_low"] == before["ci_low"]
    s = client.get("/api/status").json()
    assert s["rounds"]["2"]["n_rep_rows"] == 6  # original generation intact


def test_successful_swap_publishes_new_generation(app_env, paths, tmp_path):
    client, data_dir = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)
    gen2_before = client.get("/api/status").json()["rounds"]["2"]["generation"]

    # New, independently valid round-2 dataset (only district A samples).
    new_csv = tmp_path / "r2_new.csv"
    write_csv(new_csv, [("u1", "A", 0, 5.0), ("u2", "A", 1, 5.0)])
    new_pq = tmp_path / "r2_new.parquet"
    write_parquet(new_pq, [
        ("u1", "rep1", 3.0), ("u2", "rep1", 7.0),
        ("u1", "rep2", 9.0), ("u2", "rep2", 1.0),
    ])
    r = upload(client, 2, new_csv, new_pq)
    assert r.status_code == 200
    gen2_after = r.json()["generation"]
    assert gen2_after > gen2_before
    # Old generation tables were dropped (refcount zero at swap time).
    import duckdb
    con = duckdb.connect(str(data_dir / "bus_survey.duckdb"))
    try:
        names = {row[0] for row in con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()}
    finally:
        con.close()
    assert f"s_gen_{gen2_after}" in names and f"r_gen_{gen2_after}" in names
    assert f"s_gen_{gen2_before}" not in names
    d = client.get("/api/result", params={"district": "A"}).json()
    # new r2 main proportion = 5/10 = 0.5; r1 stays 0.4 -> diff 0.1
    assert d["point_estimate"] == pytest.approx(0.1)
    # rep diffs: rep1 0.7-0.8=-0.1; rep2 0.1-0=0.1 -> type-7 q = -0.095 / 0.095
    assert d["ci_low"] == pytest.approx(-0.095)
    assert d["ci_high"] == pytest.approx(0.095)


def test_snapshot_pins_old_generation_during_query(app_env, paths, tmp_path):
    """acquire/release snapshot keeps a swapped generation alive for the query."""
    from app import db
    client, _ = app_env
    r1_csv, r1_pq, r2_csv, r2_pq = paths
    upload(client, 1, r1_csv, r1_pq)
    upload(client, 2, r2_csv, r2_pq)

    gen2 = db.current_generation(2)
    pinned = db.acquire_snapshot(2)
    assert pinned == gen2
    # Swap round 2 while the "long query" holds its snapshot.
    new_csv = tmp_path / "r2_new.csv"
    write_csv(new_csv, [("u1", "A", 0, 5.0), ("u2", "A", 1, 5.0)])
    new_pq = tmp_path / "r2_new.parquet"
    write_parquet(new_pq, [
        ("u1", "rep1", 3.0), ("u2", "rep1", 7.0),
        ("u1", "rep2", 9.0), ("u2", "rep2", 1.0),
    ])
    assert upload(client, 2, new_csv, new_pq).status_code == 200
    assert db.current_generation(2) != gen2

    import duckdb
    con = duckdb.connect(str(db.DB_PATH))
    try:
        names = {row[0] for row in con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()}
    finally:
        con.close()
    # Pinned old generation must still be readable.
    assert f"s_gen_{gen2}" in names and f"r_gen_{gen2}" in names
    db.release_snapshot(gen2)
    db.drop_if_unreferenced(gen2)
    con = duckdb.connect(str(db.DB_PATH))
    try:
        names = {row[0] for row in con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()}
    finally:
        con.close()
    assert f"s_gen_{gen2}" not in names
