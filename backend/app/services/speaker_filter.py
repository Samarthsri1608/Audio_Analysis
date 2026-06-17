"""
speaker_filter.py
Local speaker diarization using:
  1. WebRTC VAD   - fast C-based voice activity detection (no cloud calls)
  2. Parselmouth  - MFCC extraction via Praat (already installed for voice modulation)
  3. K-Means      - scipy.cluster.vq to cluster frames into 2 speaker groups

This fully replaces the previous Gemini-based diarization, cutting Gemini API usage
to a single call per interview (transcription only).

Function signature (unchanged):  extract_interviewee_audio(audio_path: str) -> str
"""

import os
import array
import logging
import warnings
import struct

import numpy as np
import webrtcvad
import parselmouth
from scipy.cluster.vq import kmeans, vq, whiten
from pydub import AudioSegment

from app.settings import settings
from app.shared_models import get_diarization_cache, get_file_id

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="std\\(\\): degrees of freedom is <= 0",
    category=UserWarning,
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16_000        # Hz  (WebRTC VAD only supports 8k, 16k, 32k, 48k)
FRAME_DURATION_MS  = 30           # ms  (10 | 20 | 30)
VAD_AGGRESSIVENESS = 2            # 0 = lenient … 3 = very aggressive
N_MFCC             = 13           # number of MFCC coefficients
MFCC_STEP_S        = 0.030        # seconds between MFCC frames (matches VAD frame)
MIN_SEGMENT_MS     = 300          # merge gaps shorter than this into the same segment
DIARIZE_SAMPLE_S   = 180          # learn speaker clusters from first N seconds only
                                  # (speakers are established in the first ~60-90s;
                                  #  sampling the first 3 min is sufficient and fast)


def _load_mono16k(audio_path: str) -> AudioSegment:
    """Load any audio file and normalise to 16 kHz 16-bit mono."""
    audio = AudioSegment.from_file(audio_path)
    return audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)


def _vad_speech_frames(audio: AudioSegment) -> list[tuple[int, bytes]]:
    """
    Run WebRTC VAD over the audio and return a list of (start_ms, frame_bytes)
    tuples for every frame classified as speech.
    """
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_len_ms = FRAME_DURATION_MS
    # Number of bytes per frame: sample_rate * 2 bytes/sample * frame_duration
    bytes_per_frame = int(SAMPLE_RATE * 2 * frame_len_ms / 1000)

    raw = audio.raw_data
    speech_frames: list[tuple[int, bytes]] = []

    for start in range(0, len(raw) - bytes_per_frame + 1, bytes_per_frame):
        frame = raw[start : start + bytes_per_frame]
        start_ms = (start // 2) * 1000 // SAMPLE_RATE   # bytes → ms
        if vad.is_speech(frame, SAMPLE_RATE):
            speech_frames.append((start_ms, frame))

    logger.info(f"VAD: {len(speech_frames)} speech frames out of "
                f"{len(raw) // bytes_per_frame} total frames.")
    return speech_frames


def _extract_mfccs(
    audio_path: str,
    end_time_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract MFCC feature matrix using Praat (parselmouth).

    Args:
        audio_path  — path to the audio file
        end_time_s  — if set, only extract MFCCs up to this time in seconds
                      (used to learn cluster centroids from a short sample)
    Returns:
        mfcc_matrix  — shape (n_frames, N_MFCC)
        frame_times  — centre time in seconds for each MFCC frame
    """
    sound = parselmouth.Sound(audio_path)
    # Trim to sample window if requested (saves significant CPU on long files)
    if end_time_s is not None and end_time_s < sound.duration:
        sound = sound.extract_part(
            from_time=0.0,
            to_time=end_time_s,
            preserve_times=False,
        )
    mfcc_obj = sound.to_mfcc(
        number_of_coefficients=N_MFCC,
        window_length=0.025,   # 25 ms analysis window
        time_step=MFCC_STEP_S,
        firstFilterFreqency=100.0,
        distance_between_filters=100.0,
    )
    # to_array() returns shape (N_MFCC, n_frames) — transpose
    matrix = mfcc_obj.to_array().T          # → (n_frames, N_MFCC)
    n_frames = matrix.shape[0]
    xs = mfcc_obj.x1 + np.arange(n_frames) * mfcc_obj.dx
    return matrix, xs  # xs = frame centre times in seconds


def _kmeans_speaker_labels(mfcc_matrix: np.ndarray) -> np.ndarray:
    """
    Cluster MFCC frames into 2 speakers using scipy K-Means.
    Returns integer label array (0 or 1) with length == n_frames.
    """
    # Whiten (standardise variance per feature) for better K-Means convergence
    whitened = whiten(mfcc_matrix)

    # scipy kmeans needs the data to be finite; drop any NaN rows
    valid_mask = np.all(np.isfinite(whitened), axis=1)
    valid_data = whitened[valid_mask]

    if valid_data.shape[0] < 2:
        # Degenerate case — mark everything as speaker 0
        return np.zeros(mfcc_matrix.shape[0], dtype=int)

    codebook, _ = kmeans(valid_data, 2, iter=20)
    codes_valid, _ = vq(valid_data, codebook)

    # Re-insert labels for NaN rows (assign label 0)
    labels = np.zeros(mfcc_matrix.shape[0], dtype=int)
    labels[valid_mask] = codes_valid
    return labels


def _identify_candidate_label(
    labels: np.ndarray,
    frame_times: np.ndarray,
) -> int:
    """
    The candidate (interviewee) speaks the majority of the time.
    Return the label (0 or 1) with the longest total duration.
    """
    dur_0 = np.sum(labels == 0) * MFCC_STEP_S
    dur_1 = np.sum(labels == 1) * MFCC_STEP_S
    logger.info(f"Speaker-0 duration ≈ {dur_0:.1f}s | Speaker-1 duration ≈ {dur_1:.1f}s")
    candidate_label = 0 if dur_0 >= dur_1 else 1
    logger.info(f"Identified candidate as Speaker-{candidate_label}")
    return candidate_label


def _frames_to_segments(
    is_candidate_frame: np.ndarray,
    frame_times: np.ndarray,
) -> list[tuple[float, float]]:
    """
    Convert a boolean mask over MFCC frames into (start_s, end_s) segments,
    merging gaps shorter than MIN_SEGMENT_MS.
    """
    if len(frame_times) == 0:
        return []

    half = MFCC_STEP_S / 2.0
    segments: list[tuple[float, float]] = []
    in_seg = False
    seg_start = 0.0

    for i, is_cand in enumerate(is_candidate_frame):
        t = frame_times[i]
        if is_cand and not in_seg:
            seg_start = max(0.0, t - half)
            in_seg = True
        elif not is_cand and in_seg:
            seg_end = t + half
            segments.append((seg_start, seg_end))
            in_seg = False

    if in_seg:
        segments.append((seg_start, frame_times[-1] + half))

    # Merge small gaps
    min_gap_s = MIN_SEGMENT_MS / 1000.0
    merged: list[tuple[float, float]] = []
    for seg in segments:
        if merged and (seg[0] - merged[-1][1]) < min_gap_s:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    return merged


def extract_interviewee_audio(audio_path: str) -> str:
    """
    Local speaker diarization pipeline:
      1. Load + normalise audio to 16 kHz mono.
      2. WebRTC VAD — identify speech vs. silence frames.
      3. Parselmouth MFCC extraction on the full audio.
      4. K-Means (K=2) clustering to separate 2 speakers.
      5. Identify the candidate (longest speaking time).
      6. Splice candidate segments → save to *_interviewee.wav.
      7. Cache a word-level transcript stub so asr_service can use
         the Gemini transcription path on the candidate-only file.

    Returns the path to the candidate-only WAV file.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"[Local Diarization] Loading audio: {audio_path}")
    audio_16k = _load_mono16k(audio_path)
    total_duration_s = len(audio_16k) / 1000.0
    logger.info(f"[Local Diarization] Duration: {total_duration_s:.1f}s")

    # ── Step 1: WebRTC VAD ────────────────────────────────────────────────────
    speech_frames = _vad_speech_frames(audio_16k)
    if not speech_frames:
        raise ValueError("WebRTC VAD detected no speech in the audio file.")

    # Build a speech-only audio segment for a quick sanity check
    speech_pct = len(speech_frames) * FRAME_DURATION_MS / len(audio_16k) * 100
    logger.info(f"[Local Diarization] Speech occupancy: {speech_pct:.1f}%")

    # ── Step 2: MFCC on SHORT SAMPLE to learn cluster centroids ───────────────
    # We only need the first DIARIZE_SAMPLE_S seconds to identify the two
    # speakers reliably — they introduce themselves in the first ~60 s.
    # This avoids parselmouth processing a full 45-minute file.
    sample_end = min(total_duration_s, float(DIARIZE_SAMPLE_S))
    logger.info(
        f"[Local Diarization] Extracting MFCCs from first {sample_end:.0f}s "
        f"(of {total_duration_s:.0f}s total) for cluster training..."
    )
    sample_mfcc, sample_times = _extract_mfccs(audio_path, end_time_s=sample_end)

    if sample_mfcc.shape[0] < 4:
        raise ValueError("Audio too short for MFCC extraction.")

    # ── Step 3: K-Means on sample → learn codebook ─────────────────────────────
    logger.info("[Local Diarization] Running K-Means (K=2) on sample to learn speaker clusters...")
    from scipy.cluster.vq import whiten, kmeans, vq
    whitened_sample = whiten(sample_mfcc)
    valid_mask_s = np.all(np.isfinite(whitened_sample), axis=1)
    valid_sample = whitened_sample[valid_mask_s]
    if valid_sample.shape[0] < 2:
        raise ValueError("Not enough valid MFCC frames in sample for clustering.")
    codebook, _ = kmeans(valid_sample, 2, iter=20)

    # ── Step 3b: Label FULL audio using trained codebook ───────────────────────
    logger.info("[Local Diarization] Labelling full audio with trained speaker codebook...")
    full_mfcc, frame_times = _extract_mfccs(audio_path)   # full duration
    whitened_full = whiten(full_mfcc)
    valid_mask_f = np.all(np.isfinite(whitened_full), axis=1)
    labels = np.zeros(full_mfcc.shape[0], dtype=int)
    codes_valid, _ = vq(whitened_full[valid_mask_f], codebook)
    labels[valid_mask_f] = codes_valid

    # ── Step 4: Identify candidate ─────────────────────────────────────────────
    candidate_label = _identify_candidate_label(labels, frame_times)
    is_candidate = (labels == candidate_label)

    # ── Step 5: Convert frames → time segments ─────────────────────────────────
    candidate_segments = _frames_to_segments(is_candidate, frame_times)
    logger.info(f"[Local Diarization] Candidate segments: {len(candidate_segments)}")

    if not candidate_segments:
        raise ValueError("No candidate speech segments found after clustering.")

    # ── Step 6: Splice candidate audio ─────────────────────────────────────────
    # Reload original audio (may be stereo / any rate) for clean splicing
    orig_audio = AudioSegment.from_file(audio_path)
    interviewee_audio = AudioSegment.empty()
    reconstructed_segments: list[dict] = []
    current_time_ms = 0.0

    for start_s, end_s in candidate_segments:
        start_ms = max(0, int(start_s * 1000))
        end_ms   = min(len(orig_audio), int(end_s * 1000))
        if end_ms <= start_ms:
            continue

        chunk = orig_audio[start_ms:end_ms]
        interviewee_audio += chunk
        dur_ms = end_ms - start_ms

        # Build a lightweight segment record (text will be filled by ASR)
        reconstructed_segments.append({
            "id":    len(reconstructed_segments),
            "start": current_time_ms / 1000.0,
            "end":   (current_time_ms + dur_ms) / 1000.0,
            "text":  "",
            "words": [],
        })
        current_time_ms += dur_ms

    if len(interviewee_audio) == 0:
        raise ValueError("No candidate audio extracted after splicing.")

    # ── Step 7: Save interviewee audio ─────────────────────────────────────────
    base, _ = os.path.splitext(audio_path)
    output_path = f"{base}_interviewee.wav"
    interviewee_audio.export(output_path, format="wav")
    logger.info(
        f"[Local Diarization] Interviewee audio saved: {output_path} "
        f"({len(interviewee_audio)/1000:.1f}s)"
    )

    # ── Step 8: Cache empty transcript stub (ASR will fill it) ────────────────
    # We leave text empty so asr_service.transcribe_audio will run Gemini ASR
    # on the candidate-only file (one Gemini call total, not two).
    file_id = get_file_id(output_path)
    # Do NOT pre-populate cache — let transcribe_audio do its job on the
    # clean candidate-only audio via Gemini so timestamps are accurate.
    logger.info(f"[Local Diarization] File ID for ASR: {file_id}")

    return output_path
