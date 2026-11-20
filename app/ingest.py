"""Upload validation and atomic generation switching.

A round is ingested as a pair of files:

* sample CSV:        sample_id, district, bus, main_weight
* replicate Parquet: sample_id, rep_id, weight

All checks run inside one transaction against staging tables.  On any
failure the transaction is rolled back and the previously visible data is
left untouched; on success the staging tables are atomically renamed into a
new generation and published via ``meta_current``.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import duckdb

from . import db

CSV_COLUMNS = ["sample_id", "district", "bus", "main_weight"]
PARQUET_COLUMNS = ["sample_id", "rep_id", "weight"]


class IngestionError(Exception):
    """A validation failure; maps to HTTP 422."""


def _column_names(con: duckdb.DuckDBPyConnection, reader_sql: str) -> list[str]:
    try:
        return [d[0].lower() for d in con.execute(f"SELECT * FROM {reader_sql} LIMIT 0").description]
    except Exception as exc:  # unreadable / corrupt file
        raise IngestionError(f"无法读取文件: {exc}") from exc


def _check_columns(actual: list[str], expected: list[str], label: str) -> None:
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    problems = []
    if missing:
        problems.append(f"缺少列: {', '.join(missing)}")
    if extra:
        problems.append(f"多余列: {', '.join(extra)}")
    if problems:
        raise IngestionError(f"{label}列结构不匹配（要求 {', '.join(expected)}）；" + "；".join(problems))


def ingest_round(
    round_no: int,
    sample_csv_path: str,
    replicate_parquet_path: str,
) -> dict:
    """Validate one round's files and publish them as a new generation."""
    if round_no not in (1, 2):
        raise IngestionError("轮次必须是 1 或 2")
    if not Path(sample_csv_path).is_file():
        raise IngestionError("样本 CSV 文件不存在")
    if not Path(replicate_parquet_path).is_file():
        raise IngestionError("重抽权重 Parquet 文件不存在")

    with db.writer_lock():
        con = db._check()
        # Filenames here are server-side temp paths created by the endpoint,
        # never user-supplied identifiers.
        csv_reader = f"read_csv_auto({_sql_string(sample_csv_path)})"
        pq_reader = f"read_parquet({_sql_string(replicate_parquet_path)})"

        csv_cols = _column_names(con, csv_reader)
        _check_columns(csv_cols, CSV_COLUMNS, "样本 CSV")
        pq_cols = _column_names(con, pq_reader)
        _check_columns(pq_cols, PARQUET_COLUMNS, "重抽权重 Parquet")

        stage_s = f"stage_s_{os.getpid()}_{os.urandom(4).hex()}"
        stage_r = f"stage_r_{os.getpid()}_{os.urandom(4).hex()}"
        con.execute("BEGIN TRANSACTION")
        try:
            # Explicit casts: bad values surface as IngestionError, not
            # implicit type quirks downstream.
            try:
                con.execute(
                    f"""
                    CREATE TABLE "{stage_s}" AS
                    SELECT CAST(sample_id AS VARCHAR)  AS sample_id,
                           CAST(district AS VARCHAR)  AS district,
                           CAST(bus AS INTEGER)       AS bus,
                           CAST(main_weight AS DOUBLE) AS main_weight
                    FROM {csv_reader}
                    """
                )
            except Exception as exc:
                raise IngestionError(f"样本 CSV 类型转换失败（bus 须为 0/1，main_weight 须为数值）: {exc}") from exc
            try:
                con.execute(
                    f"""
                    CREATE TABLE "{stage_r}" AS
                    SELECT CAST(sample_id AS VARCHAR) AS sample_id,
                           CAST(rep_id AS VARCHAR)    AS rep_id,
                           CAST(weight AS DOUBLE)     AS weight
                    FROM {pq_reader}
                    """
                )
            except Exception as exc:
                raise IngestionError(f"重抽权重 Parquet 类型转换失败（weight 须为数值）: {exc}") from exc

            _validate_samples(con, stage_s)
            _validate_replicates(con, stage_s, stage_r)

            stats = con.execute(
                f"""
                SELECT (SELECT count(*) FROM "{stage_s}"),
                       (SELECT count(DISTINCT rep_id) FROM "{stage_r}"),
                       (SELECT count(*) FROM "{stage_r}")
                """
            ).fetchone()
            n_samples, n_replicates, n_rep_rows = (int(x) for x in stats)
            con.execute("COMMIT")
        except Exception:
            try:
                con.execute("ROLLBACK")
            except Exception:
                pass
            _safe_drop(stage_s)
            _safe_drop(stage_r)
            raise

        try:
            generation = db.commit_generation(
                round_no, stage_s, stage_r, n_samples, n_replicates, n_rep_rows
            )
        except Exception:
            _safe_drop(stage_s)
            _safe_drop(stage_r)
            raise

    return {
        "round": round_no,
        "generation": generation,
        "n_samples": n_samples,
        "n_replicates": n_replicates,
        "n_rep_rows": n_rep_rows,
    }


def _validate_samples(con: duckdb.DuckDBPyConnection, stage_s: str) -> None:
    n = con.execute(f'SELECT count(*) FROM "{stage_s}"').fetchone()[0]
    if n == 0:
        raise IngestionError("样本 CSV 为空，至少需要一个样本")

    if con.execute(
        f"""SELECT count(*) FROM "{stage_s}"
            WHERE sample_id IS NULL OR trim(sample_id) = ''
               OR district IS NULL OR trim(district) = ''"""
    ).fetchone()[0] > 0:
        raise IngestionError("存在样本号或区为空的记录")

    bad_bus = con.execute(
        f'SELECT count(*) FROM "{stage_s}" WHERE bus NOT IN (0, 1) OR bus IS NULL'
    ).fetchone()[0]
    if bad_bus:
        raise IngestionError(f"是否乘公交(bus) 只能取 0 或 1，发现 {bad_bus} 条违规记录")

    bad_w = con.execute(
        f"""SELECT count(*) FROM "{stage_s}"
            WHERE main_weight IS NULL OR NOT isfinite(main_weight) OR main_weight < 0"""
    ).fetchone()[0]
    if bad_w:
        raise IngestionError(f"主权重必须为有限非负数，发现 {bad_w} 条违规记录")

    dup = con.execute(
        f"SELECT count(*) FROM (SELECT sample_id FROM \"{stage_s}\" GROUP BY 1 HAVING count(*) > 1)"
    ).fetchone()[0]
    if dup:
        raise IngestionError(f"样本号在样本 CSV 中重复，涉及 {dup} 个样本号")


def _validate_replicates(
    con: duckdb.DuckDBPyConnection, stage_s: str, stage_r: str
) -> None:
    n = con.execute(f'SELECT count(*) FROM "{stage_r}"').fetchone()[0]
    if n == 0:
        raise IngestionError("重抽权重 Parquet 为空")

    if con.execute(
        f"""SELECT count(*) FROM "{stage_r}"
            WHERE sample_id IS NULL OR trim(sample_id) = ''
               OR rep_id IS NULL OR trim(rep_id) = ''"""
    ).fetchone()[0] > 0:
        raise IngestionError("存在样本号或重抽编号为空的权重记录")

    bad_w = con.execute(
        f"""SELECT count(*) FROM "{stage_r}"
            WHERE weight IS NULL OR NOT isfinite(weight) OR weight < 0"""
    ).fetchone()[0]
    if bad_w:
        raise IngestionError(f"重抽权重必须为有限非负数，发现 {bad_w} 条违规记录")

    # A replicate whose weights sum to zero can never form a proportion.
    zero_reps = con.execute(
        f"""SELECT count(*) FROM (
            SELECT rep_id FROM "{stage_r}" GROUP BY rep_id HAVING sum(weight) = 0
        )"""
    ).fetchone()[0]
    if zero_reps:
        raise IngestionError(f"存在总权重为零的重抽编号，涉及 {zero_reps} 个重抽（无法计算比例）")

    # Replicate weights may only reference samples of this very round.
    foreign = con.execute(
        f"""
        SELECT count(*) FROM "{stage_r}" r
        WHERE NOT EXISTS (SELECT 1 FROM "{stage_s}" s WHERE s.sample_id = r.sample_id)
        """
    ).fetchone()[0]
    if foreign:
        raise IngestionError(
            f"重抽权重引用了本轮样本 CSV 中不存在的样本号，涉及 {foreign} 条记录"
        )

    # Exactly one weight per (sample, replicate): zero rows are missing,
    # more than one means duplicates.
    missing = con.execute(
        f"""
        WITH reps AS (SELECT DISTINCT rep_id FROM "{stage_r}"),
             samples AS (SELECT sample_id FROM "{stage_s}")
        SELECT count(*)
        FROM reps CROSS JOIN samples x
        WHERE NOT EXISTS (
            SELECT 1 FROM "{stage_r}" r
            WHERE r.rep_id = reps.rep_id AND r.sample_id = x.sample_id
        )
        """
    ).fetchone()[0]
    if missing:
        raise IngestionError(
            "每个样本在每次重抽中须恰有一条权重："
            f"发现 {missing} 个 (样本, 重抽) 组合缺少权重"
        )
    duplicated = con.execute(
        f"""
        SELECT count(*) FROM (
            SELECT sample_id, rep_id FROM "{stage_r}"
            GROUP BY 1, 2 HAVING count(*) > 1
        )
        """
    ).fetchone()[0]
    if duplicated:
        raise IngestionError(
            "每个样本在每次重抽中须恰有一条权重："
            f"发现 {duplicated} 个 (样本, 重抽) 组合存在重复权重"
        )


def _safe_drop(table: str) -> None:
    try:
        db._check().execute(f'DROP TABLE IF EXISTS "{table}"')
    except Exception:
        pass


def _sql_string(value: str) -> str:
    """Quote a server-side path as a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def save_upload(upload_dir: str, filename: str, fileobj) -> str:
    """Stream an uploaded file object into a fresh temp directory.

    Chunked copy keeps large uploads out of memory.
    """
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()[:16]
    fd, path = tempfile.mkstemp(suffix=suffix, dir=upload_dir)
    try:
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(fileobj, out, length=1024 * 1024)
    except Exception:
        os.unlink(path)
        raise
    return path


def remove_upload_dir(upload_dir: str) -> None:
    shutil.rmtree(upload_dir, ignore_errors=True)
