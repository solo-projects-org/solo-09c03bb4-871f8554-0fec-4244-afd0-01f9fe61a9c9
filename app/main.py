"""FastAPI application: upload rounds, inspect status, query districts."""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from . import analysis, db, ingest

UPLOAD_ROOT = Path(os.environ.get("UPLOAD_TMP_DIR", str(db.TMP_DIR)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    db.init_db()
    yield


app = FastAPI(title="公交出行比例两轮重抽分析", version="1.0.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
def status() -> dict:
    payload = {"rounds": {}}
    for round_no in (1, 2):
        gen = db.current_generation(round_no)
        if gen is None:
            payload["rounds"][str(round_no)] = {"loaded": False}
            continue
        info = db.generation_info(gen)
        # info: round, created_at, n_samples, n_replicates, n_rep_rows
        payload["rounds"][str(round_no)] = {
            "loaded": True,
            "generation": gen,
            "created_at": info[1].isoformat() if info else None,
            "n_samples": info[2],
            "n_replicates": info[3],
            "n_rep_rows": info[4],
        }
    payload["districts"] = analysis.list_districts()
    return payload


@app.post("/api/rounds/{round_no}")
def upload_round(
    round_no: int,
    sample_csv: UploadFile = File(...),
    replicate_parquet: UploadFile = File(...),
) -> JSONResponse:
    # Sync endpoint: Starlette runs it in the worker thread, and uploads are
    # streamed straight to disk instead of being buffered in memory.
    if round_no not in (1, 2):
        raise HTTPException(status_code=422, detail="轮次必须是 1 或 2")
    if not sample_csv.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="样本文件必须是 .csv")
    if not replicate_parquet.filename.lower().endswith(".parquet"):
        raise HTTPException(status_code=422, detail="重抽权重文件必须是 .parquet")

    upload_dir = tempfile.mkdtemp(prefix=f"round{round_no}_", dir=str(UPLOAD_ROOT))
    try:
        csv_path = ingest.save_upload(upload_dir, sample_csv.filename, sample_csv.file)
        pq_path = ingest.save_upload(
            upload_dir, replicate_parquet.filename, replicate_parquet.file
        )
        try:
            result = ingest.ingest_round(round_no, csv_path, pq_path)
        except ingest.IngestionError as exc:
            # Failed replacement: previously published data remains queryable.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        ingest.remove_upload_dir(upload_dir)

    return JSONResponse({"ok": True, **result})


@app.get("/api/districts")
def districts() -> dict:
    return {"districts": analysis.list_districts()}


# GET with a query parameter is the natural fit for the browser UI.
@app.get("/api/result")
def result(district: str) -> dict:
    district = (district or "").strip()
    if not district:
        raise HTTPException(status_code=422, detail="请选择一个区")
    try:
        return analysis.compute(district)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
