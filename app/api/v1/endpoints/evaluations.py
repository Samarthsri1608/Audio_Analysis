import json
import logging
import httpx
import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.api.v1.schemas.evaluations import TrainRequest
from app.services.evaluation_service import (
    run_evaluation,
    fetch_recording_urls,
    pick_best_url,
    cache_get,
    cache_put,
)
from app.services.report_generator import generate_report
from app.services.scoring_engine.train_model import train_scoring_models

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/train")
async def train_models(request: TrainRequest):
    try:
        results = train_scoring_models(request.dataset_path)
        return {"message": "Models trained successfully", "success": True, "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate_audio(file: UploadFile = File(...)):
    """
    Debug endpoint — returns the full raw evaluation data.
    Accepts .wav, .mp3, .mp4.
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".mp4")):
        raise HTTPException(status_code=400, detail="Unsupported file format. Accepted: .wav, .mp3, .mp4")
    try:
        result = await run_evaluation(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message":             f"Successfully evaluated {result['filename']}.",
        "success":             True,
        "data": {
            "preprocessing_flags": result["preprocessing_flags"],
            "duration_ms":         result["duration_ms"],
            "features":            result["features"],
            "scores":              result["scores"],
        },
    }


@router.post("/report")
async def generate_candidate_report(file: UploadFile = File(...)):
    """
    Client-facing endpoint — scores + Gemini-powered reasoning.
    Accepts .wav, .mp3, .mp4.
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".mp4")):
        raise HTTPException(status_code=400, detail="Unsupported file format. Accepted: .wav, .mp3, .mp4")
    try:
        result = await run_evaluation(file)
        report = generate_report(
            scores=result["scores"],
            features=result["features"],
            filename=result["filename"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Report generated.", "success": True, "data": report}


@router.post("/evaluate_by_id")
async def evaluate_by_candidate_id(response_id: str):
    """
    Streaming evaluation for a candidate identified by their response ID.
    Results are cached by response_id — identical calls return the same
    report (idempotent). Cache TTL is 7 days.
    """
    async def event_generator():
        try:
            # Check Cache
            cached = cache_get(response_id)
            if cached:
                yield json.dumps({"type": "status", "message": "Result retrieved from cache."}) + "\n"
                yield json.dumps({"type": "result",  "payload": cached}) + "\n"
                return

            yield json.dumps({"type": "status", "message": "Connecting to interview vault..."}) + "\n"

            links = await fetch_recording_urls(response_id)
            yield json.dumps({"type": "status", "message": f"Found {len(links)} recording(s). Selecting primary..."}) + "\n"

            chosen_url = await pick_best_url(links)
            filename   = chosen_url.split("?")[0].split("/")[-1]
            if not filename.lower().endswith(".mp4"):
                filename += ".mp4"

            yield json.dumps({"type": "status", "message": "Downloading recording..."}) + "\n"

            async with httpx.AsyncClient(timeout=120) as client:
                dl = await client.get(chosen_url)
                dl.raise_for_status()

            video_bytes = io.BytesIO(dl.content)
            upload      = StarletteUploadFile(filename=filename, file=video_bytes)

            yield json.dumps({"type": "status", "message": "Running evaluation pipeline..."}) + "\n"

            result = await run_evaluation(upload)
            report = generate_report(
                scores=result["scores"],
                features=result["features"],
                filename=result["filename"],
            )

            # Cache write
            cache_put(response_id, report)

            yield json.dumps({"type": "result", "payload": report}) + "\n"

        except HTTPException as e:
            yield json.dumps({"type": "error", "message": e.detail}) + "\n"
        except Exception as e:
            logger.error(f"evaluate_by_id failed: {e}")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
