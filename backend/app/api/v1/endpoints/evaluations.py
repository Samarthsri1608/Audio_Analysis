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
        import os
        import tempfile
        import shutil
        import asyncio

        temp_dir = tempfile.mkdtemp(prefix="video_concat_")
        downloaded_files = []

        try:
            # Check Cache
            cached = cache_get(response_id)
            if cached:
                yield json.dumps({"type": "status", "message": "Result retrieved from cache."}) + "\n"
                yield json.dumps({"type": "result",  "payload": cached}) + "\n"
                return

            yield json.dumps({"type": "status", "message": "Connecting to interview vault..."}) + "\n"

            # Run loop for up to 25 questions to fetch recordings
            for q_no in range(1, 26):
                url = f"https://interview-api.zeko.ai/dashboard/api/v2/report/recordings/question?responseId={response_id}&quesNo={q_no}"
                try:
                    async with httpx.AsyncClient(timeout=15) as client:
                        resp = await client.get(url)
                        if resp.status_code != 200:
                            logger.info(f"Stop loop at Q{q_no}: API returned status {resp.status_code}")
                            break
                        resp_data = resp.json()
                        data = resp_data.get("data")
                        if not data or not isinstance(data, dict):
                            logger.info(f"Stop loop at Q{q_no}: No data dictionary in response")
                            break
                        recording_link = data.get("recordingLink")
                        if not recording_link:
                            logger.info(f"Stop loop at Q{q_no}: No recordingLink in data")
                            break

                    ext = ".mp4"
                    if "?" in recording_link:
                        url_path = recording_link.split("?")[0]
                    else:
                        url_path = recording_link
                    if url_path.endswith((".mp4", ".webm", ".wav", ".mp3", ".ogg")):
                        ext = os.path.splitext(url_path)[1]

                    local_path = os.path.join(temp_dir, f"q_{q_no}{ext}")
                    yield json.dumps({"type": "status", "message": f"Downloading recording for Question {q_no}..."}) + "\n"

                    async with httpx.AsyncClient(timeout=120) as client:
                        dl = await client.get(recording_link)
                        dl.raise_for_status()

                    with open(local_path, "wb") as f:
                        f.write(dl.content)

                    downloaded_files.append(local_path)
                except Exception as e:
                    logger.info(f"Break loop for response_id {response_id} at Q{q_no}: {e}")
                    break

            if not downloaded_files:
                raise HTTPException(status_code=404, detail=f"No recordings found for responseId '{response_id}'.")

            # Merge downloaded videos
            yield json.dumps({"type": "status", "message": f"Merging {len(downloaded_files)} recording(s)..."}) + "\n"

            if len(downloaded_files) == 1:
                merged_video_path = downloaded_files[0]
            else:
                first_ext = os.path.splitext(downloaded_files[0])[1] or ".mp4"
                merged_video_path = os.path.join(temp_dir, f"merged{first_ext}")
                list_file_path = os.path.join(temp_dir, "files.txt")
                with open(list_file_path, "w") as lf:
                    for filepath in downloaded_files:
                        lf.write(f"file '{filepath}'\n")

                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file_path,
                    "-c", "copy",
                    merged_video_path,
                    "-y",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(f"ffmpeg video concat failed (exit {proc.returncode}): {stderr.decode()}")

            filename = os.path.basename(merged_video_path)

            with open(merged_video_path, "rb") as f:
                video_bytes = io.BytesIO(f.read())
            upload = StarletteUploadFile(filename=filename, file=video_bytes)

            yield json.dumps({"type": "status", "message": "Running evaluation pipeline (skipping diarization)..."}) + "\n"

            result = await run_evaluation(upload, skip_diarization=True)
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
        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
