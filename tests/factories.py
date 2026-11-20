"""Synthetic dataset generation for tests (small hand-built + large scale)."""
from __future__ import annotations

import csv as csv_mod

import duckdb


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["sample_id", "district", "bus", "main_weight"])
        writer.writerows(rows)


def write_parquet(path, rows):
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TEMP TABLE r (sample_id VARCHAR, rep_id VARCHAR, weight DOUBLE)"
        )
        con.executemany("INSERT INTO r VALUES (?, ?, ?)", list(rows))
        con.execute("COPY r TO ? (FORMAT PARQUET)", [str(path)])
    finally:
        con.close()


def hand_built(tmp: Path):
    """Two tiny rounds with hand-computable answers, reps rep1/rep2."""
    # Round 1, district A: s1(bus1,w4), s2(bus0,w6); district B: s3(bus1,w2)
    r1_csv = tmp / "r1.csv"
    write_csv(r1_csv, [
        ("s1", "A", 1, 4.0),
        ("s2", "A", 0, 6.0),
        ("s3", "B", 1, 2.0),
    ])
    # Round 2, district A: t1(bus1,w9), t2(bus1,w1); B: t3(bus0,w5)
    r2_csv = tmp / "r2.csv"
    write_csv(r2_csv, [
        ("t1", "A", 1, 9.0),
        ("t2", "A", 1, 1.0),
        ("t3", "B", 0, 5.0),
    ])

    # Replicate weights (full coverage: one row per sample per rep).
    r1_pq = tmp / "r1.parquet"
    write_parquet(r1_pq, [
        ("s1", "rep1", 8.0), ("s2", "rep1", 2.0), ("s3", "rep1", 2.0),
        ("s1", "rep2", 0.0), ("s2", "rep2", 10.0), ("s3", "rep2", 2.0),
    ])
    r2_pq = tmp / "r2.parquet"
    write_parquet(r2_pq, [
        ("t1", "rep1", 5.0), ("t2", "rep1", 5.0), ("t3", "rep1", 5.0),
        ("t1", "rep2", 6.0), ("t2", "rep2", 4.0), ("t3", "rep2", 5.0),
    ])
    return r1_csv, r1_pq, r2_csv, r2_pq


def upload(client, round_no: int, csv_path: Path, pq_path: Path):
    with open(csv_path, "rb") as f1, open(pq_path, "rb") as f2:
        return client.post(
            f"/api/rounds/{round_no}",
            files={
                "sample_csv": (csv_path.name, f1.read(), "text/csv"),
                "replicate_parquet": (pq_path.name, f2.read(), "application/octet-stream"),
            },
        )


# ---------------------------------------------------------------- large data
def make_large(out_dir: Path, n_samples: int, n_reps: int, seed: int = 7):
    """Generate one round's CSV + Parquet with n_samples * n_reps weight rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE samples AS "
                    "SELECT 's' || i AS sample_id, "
                    "'D' || (i % 12) AS district, "
                    "(abs(hash(i + ?)) % 2)::INTEGER AS bus, "
                    "round(1 + (abs(hash(i * 3 + ?)) % 1000) / 100.0, 2) AS main_weight "
                    "FROM range(?) t(i)", [seed, seed, n_samples])
        csv_path = out_dir / "samples.csv"
        con.execute("COPY samples TO ? (HEADER, DELIMITER ',')", [str(csv_path)])
        pq_path = out_dir / "replicates.parquet"
        con.execute(
            """
            COPY (
                SELECT s.sample_id,
                       'rep' || lpad(k::VARCHAR, 5, '0') AS rep_id,
                       round(s.main_weight * (0.5 + (abs(hash(s.sample_id || '#' || k::VARCHAR || '#') + ?) % 200) / 100.0), 4) AS weight
                FROM samples s
                CROSS JOIN range(?) t(k)
            ) TO ? (FORMAT PARQUET)
            """,
            [seed, n_reps, str(pq_path)],
        )
    finally:
        con.close()
    return csv_path, pq_path
