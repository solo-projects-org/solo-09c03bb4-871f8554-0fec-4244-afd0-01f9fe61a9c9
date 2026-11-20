"""Point estimates and replicate-based difference distribution.

For a selected district and each round:

* main-weight bus proportion
      p = sum(main_weight * bus) / sum(main_weight)
* point estimate of the change: p2 - p1 (round 2 minus round 1)
* for every replicate id, the same proportion computed with that
  replicate's weights; round 1 and round 2 replicates with the same id
  form one paired difference d_k = p2_k - p1_k
* the 2.5% / 97.5% reported bounds are linear-interpolated quantiles
  (type 7, i.e. DuckDB ``quantile_cont`` on DOUBLE) of the d_k.

When the two rounds' replicate-id sets differ, or any involved weight
denominator is zero, the reason is returned and the interval is left empty.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import db

QUANTILES = (0.025, 0.975)
EXAMPLE_LIMIT = 20
ROUND_LABELS = {1: "一", 2: "二"}


@dataclass
class RoundResult:
    generation: int
    numerator: float
    denominator: float
    proportion: float | None


def list_districts() -> list[str]:
    """Districts present in either round's currently published samples."""
    con = db.read_connection()
    try:
        # Tables are generation-specific; union whatever is currently live.
        parts = []
        for round_no in (1, 2):
            gen = db.current_generation(round_no)
            if gen is not None:
                parts.append(f"SELECT district FROM \"{db.samples_table(gen)}\"")
        if not parts:
            return []
        return [
            r[0]
            for r in con.execute(" UNION ".join(parts) + " ORDER BY 1").fetchall()
        ]
    finally:
        con.close()


def _main_proportion(con, gen: int, district: str) -> RoundResult:
    table = db.samples_table(gen)
    num, den = con.execute(
        f"""
        SELECT COALESCE(sum(main_weight * bus), 0.0),
               COALESCE(sum(main_weight), 0.0)
        FROM "{table}" WHERE district = ?
        """,
        [district],
    ).fetchone()
    num, den = float(num), float(den)
    prop = None if den == 0.0 else num / den
    return RoundResult(gen, num, den, prop)


def _replicate_ids(con, gen: int) -> set[str]:
    return {
        r[0]
        for r in con.execute(
            f'SELECT DISTINCT rep_id FROM "{db.replicate_table(gen)}"'
        ).fetchall()
    }


def _replicate_proportions(con, gen: int, district: str) -> dict[str, tuple[float, float]]:
    """rep_id -> (numerator, denominator) restricted to the district."""
    s = db.samples_table(gen)
    r = db.replicate_table(gen)
    rows = con.execute(
        f"""
        SELECT r.rep_id,
               COALESCE(sum(r.weight * s.bus), 0.0) AS num,
               COALESCE(sum(r.weight), 0.0)         AS den
        FROM "{r}" r
        JOIN "{s}" s ON s.sample_id = r.sample_id
        WHERE s.district = ?
        GROUP BY r.rep_id
        """,
        [district],
    ).fetchall()
    return {rep_id: (float(num), float(den)) for rep_id, num, den in rows}


def compute(district: str) -> dict:
    if not district or not district.strip():
        raise ValueError("district must be a non-empty string")

    # Snapshot the two generations used by this query.
    gen1 = db.acquire_snapshot(1)
    gen2 = db.acquire_snapshot(2)
    try:
        missing_rounds = [r for r, g in ((1, gen1), (2, gen2)) if g is None]
        if missing_rounds:
            labels = "、".join(ROUND_LABELS[r] for r in missing_rounds)
            return {
                "district": district,
                "available": False,
                "reason": f"缺少第{labels}轮数据，无法比较两轮",
                "point_estimate": None,
                "ci_low": None,
                "ci_high": None,
            }

        # One connection for the whole query against the pinned snapshot.
        con = db.read_connection()
        try:
            r1 = _main_proportion(con, gen1, district)
            r2 = _main_proportion(con, gen2, district)

            zero_rounds = [
                ROUND_LABELS[r]
                for r, res in ((1, r1), (2, r2))
                if res.denominator == 0.0
            ]
            if zero_rounds:
                return {
                    "district": district,
                    "available": False,
                    "reason": (
                        f"区内主权重分母为零（第{'、'.join(zero_rounds)}轮），"
                        "比例差无法计算，区间留空"
                    ),
                    "point_estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "round1": _round_payload(r1),
                    "round2": _round_payload(r2),
                }

            point_estimate = r2.proportion - r1.proportion

            ids1 = _replicate_ids(con, gen1)
            ids2 = _replicate_ids(con, gen2)

            if ids1 != ids2:
                only1 = sorted(ids1 - ids2)
                only2 = sorted(ids2 - ids1)
                detail = []
                if only1:
                    detail.append("仅第一轮有: " + ", ".join(only1[:EXAMPLE_LIMIT])
                                  + (" …" if len(only1) > EXAMPLE_LIMIT else ""))
                if only2:
                    detail.append("仅第二轮有: " + ", ".join(only2[:EXAMPLE_LIMIT])
                                  + (" …" if len(only2) > EXAMPLE_LIMIT else ""))
                return {
                    "district": district,
                    "available": True,
                    "reason": "两轮重抽编号集合不同，无法按编号配对，区间留空（"
                              + "；".join(detail) + "）",
                    "point_estimate": point_estimate,
                    "ci_low": None,
                    "ci_high": None,
                    "round1": _round_payload(r1),
                    "round2": _round_payload(r2),
                    "n_replicates_round1": len(ids1),
                    "n_replicates_round2": len(ids2),
                }

            props1 = _replicate_proportions(con, gen1, district)
            props2 = _replicate_proportions(con, gen2, district)
        finally:
            con.close()

        # Zero replicate denominators: the id set matched, but a district
        # may have no positive total weight under one or more replicates.
        zero_reps = sorted(
            rep_id
            for rep_id in ids1
            if props1.get(rep_id, (0.0, 0.0))[1] == 0.0
            or props2.get(rep_id, (0.0, 0.0))[1] == 0.0
        )
        if zero_reps:
            return {
                "district": district,
                "available": True,
                "reason": (
                    "区内存在重抽权重分母为零的重抽（"
                    + ", ".join(zero_reps[:EXAMPLE_LIMIT])
                    + (" …" if len(zero_reps) > EXAMPLE_LIMIT else "")
                    + "），区间留空"
                ),
                "point_estimate": point_estimate,
                "ci_low": None,
                "ci_high": None,
                "round1": _round_payload(r1),
                "round2": _round_payload(r2),
                "n_replicates": len(ids1),
            }

        diffs = [
            props2[rep_id][0] / props2[rep_id][1]
            - props1[rep_id][0] / props1[rep_id][1]
            for rep_id in sorted(ids1)
        ]
        low, high = _quantiles(diffs)
        return {
            "district": district,
            "available": True,
            "reason": None,
            "point_estimate": point_estimate,
            "ci_low": low,
            "ci_high": high,
            "round1": _round_payload(r1),
            "round2": _round_payload(r2),
            "n_replicates": len(ids1),
        }
    finally:
        if gen1 is not None:
            db.release_snapshot(gen1)
            db.drop_if_unreferenced(gen1)
        if gen2 is not None:
            db.release_snapshot(gen2)
            db.drop_if_unreferenced(gen2)


def _round_payload(res: RoundResult) -> dict:
    return {
        "generation": res.generation,
        "weighted_bus_sum": res.numerator,
        "weight_sum": res.denominator,
        "proportion": res.proportion,
    }


def _quantiles(values: list[float]) -> tuple[float, float]:
    """Linear-interpolated (type-7) quantiles, matching DuckDB quantile_cont."""
    if not values:
        raise ValueError("no replicate differences to summarise")
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0], ordered[0]

    def at(p: float) -> float:
        h = (n - 1) * p
        lo = int(h)
        frac = h - lo
        if lo + 1 >= n:
            return ordered[-1]
        return ordered[lo] * (1.0 - frac) + ordered[lo + 1] * frac

    return at(QUANTILES[0]), at(QUANTILES[1])
