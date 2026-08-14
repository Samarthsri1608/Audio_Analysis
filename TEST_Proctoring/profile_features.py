#!/usr/bin/env python3
"""
profile_features.py — Instrument the feature extractor to find the bottleneck.

Usage (with a real WAV file, or a synthetic one):
    cd /Users/samarthsrivastava/Audio_Analysis
    .venv/bin/python TEST_Proctoring/profile_features.py [path/to/real.wav]

Outputs:
  - Per-stage wall-clock timing (always)
  - Top-20 cProfile hot-spots (always)
  - Synthetic WAV is generated if no path is given (8 seconds at 16kHz)

Stages timed:
  A. librosa.load (I/O + decode + resample)
  B. VAD / energy (voiced mask, Python loop version)
  C. Pitch tracking (pyin) — expected to be the biggest cost
  D. Speech-rate proxy
  E. Response latency
  F. Pause ratio + distribution
  G. RMS energy over voiced frames
  H. MFCCs
  I. Spectral flatness
  J. Room fingerprint + noise floor centroid
  K. Full extract_features() call (end-to-end, includes all of the above)
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.io.wavfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 16_000
DURATION_S = 8.0  # synthetic answer length


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"  [{label:42s}] {elapsed * 1000:8.1f} ms")


def make_synthetic_wav(path: str, duration_s: float = DURATION_S, sr: int = SR) -> None:
    """Generate a speech-like synthetic WAV: alternating voiced/silence segments."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = np.zeros(len(t), dtype=np.float32)
    pos = 0
    voiced_on = True
    seg_idx = 0
    while pos < len(t):
        if voiced_on:
            end = min(pos + int(sr * 0.4), len(t))
            freq = 140 + 20 * np.sin(seg_idx * 0.5)
            audio[pos:end] = 0.4 * np.sin(2 * np.pi * freq * t[pos:end])
            audio[pos:end] += 0.1 * np.sin(2 * np.pi * 2 * freq * t[pos:end])
        else:
            end = min(pos + int(sr * 0.15), len(t))
            audio[pos:end] = np.random.randn(end - pos).astype(np.float32) * 0.003
        pos = end
        voiced_on = not voiced_on
        seg_idx += 1
    scipy.io.wavfile.write(path, sr, (audio * 32767).astype(np.int16))


def profile_stages(wav_path: str) -> None:
    import librosa
    from v4_proctoring.pipeline.feature_extractor import (
        _voiced_mask_from_rms,
        _run_lengths,
        VAD_FRAME_S,
        MIN_VOICED_FRAMES,
        MIN_PAUSE_DURATION_S,
    )

    print()
    print("=" * 60)
    print(f"  Profiling: {wav_path}")
    print(f"  File size: {os.path.getsize(wav_path) / 1024:.1f} KB")
    print("=" * 60)
    print()
    print("── Per-stage wall-clock timing ──────────────────────────────")

    # ── A. librosa.load ───────────────────────────────────────────────────────
    with timer("A. librosa.load (sr=16000, mono=True)"):
        y, sr = librosa.load(wav_path, sr=16_000, mono=True)
    print(f"     → {len(y)/sr:.2f}s audio, sr={sr}")

    # ── B. VAD / energy (current vectorized) ────────────────────────────────
    with timer("B. VAD vectorized (librosa.feature.rms)"):
        voiced_mask = _voiced_mask_from_rms(y, sr)
    speech_s = voiced_mask.sum() * VAD_FRAME_S

    # ── B2. Numpy-strided comparison ─────────────────────────────────────────
    with timer("B2. VAD numpy striding (alt)"):
        frame_len = int(VAD_FRAME_S * sr)
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=frame_len)[0]
        threshold = np.percentile(rms, 60)
        voiced_mask_fast = rms > threshold

    # ── C. Pitch tracking — REMOVED (was 97% of wall time) ───────────────────
    print(f"  [{'C. pyin — REMOVED (was ~688ms, now None)':42s}]      0.0 ms [SKIPPED]")

    # ── D. Speech-rate proxy ──────────────────────────────────────────────────
    with timer("D. Speech-rate proxy"):
        srate = float(voiced_mask.sum() / (len(y) / sr)) if len(y) > 0 else None

    # ── E. Response latency ───────────────────────────────────────────────────
    with timer("E. Response latency"):
        first_voiced = int(np.argmax(voiced_mask)) if voiced_mask.any() else None
        latency = float(first_voiced * VAD_FRAME_S) if first_voiced is not None else float(len(y) / sr)

    # ── F. Pause ratio + distribution ─────────────────────────────────────────
    with timer("F. Pause ratio + distribution"):
        silence_mask = ~voiced_mask
        pause_ratio = float(silence_mask.sum() / max(len(voiced_mask), 1))
        pause_runs_s = _run_lengths(silence_mask) * VAD_FRAME_S
        meaningful_pauses = pause_runs_s[pause_runs_s >= MIN_PAUSE_DURATION_S]

    # ── G. RMS energy over voiced frames ──────────────────────────────────────
    with timer("G. RMS energy over voiced frames (Python loop)"):
        frame_hop_s = VAD_FRAME_S
        hop = frame_len
        all_energies = np.array([
            np.sqrt(np.mean(y[i: i + frame_len] ** 2))
            for i in range(0, len(y) - frame_len, hop)
        ])
        n = min(len(all_energies), len(voiced_mask))
        voiced_energies = all_energies[:n][voiced_mask[:n]]

    with timer("G. RMS energy vectorized (librosa.feature.rms)"):
        rms2 = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=frame_len)[0]
        n2 = min(len(rms2), len(voiced_mask_fast))
        voiced_energies2 = rms2[:n2][voiced_mask_fast[:n2]]

    # ── H. MFCCs — REMOVED ───────────────────────────────────────────────────
    print(f"  [{'H. MFCCs — REMOVED (was ~6ms, now None)':42s}]      0.0 ms [SKIPPED]")

    # ── I. Spectral flatness — REMOVED ────────────────────────────────────────
    print(f"  [{'I. Spectral flatness — REMOVED (was ~3.7ms)':42s}]      0.0 ms [SKIPPED]")

    # ── J. Room fingerprint + noise floor centroid ────────────────────────────
    with timer("J. Room fingerprint + noise floor centroid"):
        silence_parts = []
        in_silence = False
        run_start = 0
        for idx, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                run_start = idx
                in_silence = True
            elif not is_silent and in_silence:
                s = run_start * frame_len
                e = idx * frame_len
                silence_parts.append(y[s:e])
                in_silence = False
        if in_silence:
            silence_parts.append(y[run_start * frame_len:])
        if silence_parts:
            noise_y = np.concatenate(silence_parts)
            if len(noise_y) > sr * 0.1:
                rolloff = librosa.feature.spectral_rolloff(y=noise_y, sr=sr, roll_percent=0.85)
                room_fp = float(np.mean(rolloff))
                centroid = librosa.feature.spectral_centroid(y=noise_y, sr=sr)
                noise_centroid = float(np.mean(centroid))

    # ── K. Full extract_features() end-to-end ────────────────────────────────
    print()
    print("── Full pipeline end-to-end ──────────────────────────────────")
    from v4_proctoring.pipeline.feature_extractor import extract_features
    with timer("K. extract_features() end-to-end"):
        ev, feats = extract_features(wav_path)
    print(f"     → evaluable={ev.evaluable}, reason={ev.not_evaluable_reason}")
    if feats:
        print(f"     → latency={feats.response_latency}s, room_fp={feats.room_fingerprint}")

    # ── cProfile hot-spots ────────────────────────────────────────────────────
    print()
    print("── cProfile hot-spots (top 20 by cumulative time) ───────────")
    pr = cProfile.Profile()
    pr.enable()
    extract_features(wav_path)
    pr.disable()

    sio = io.StringIO()
    ps = pstats.Stats(pr, stream=sio).sort_stats("cumulative")
    ps.print_stats(20)
    output = sio.getvalue()
    # Only print lines that look like actual function entries
    for line in output.splitlines():
        if line.strip() and not line.startswith("   "):
            print(line)
        elif any(kw in line for kw in ["librosa", "pyin", "mfcc", "rms", "stft", "scipy", "numpy", "feature_extractor"]):
            print(line)

    print()


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else None

    if wav_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = tmp.name
        tmp.close()
        print(f"No WAV supplied — generating synthetic {DURATION_S}s audio at {SR}Hz…")
        make_synthetic_wav(wav_path)
        synthetic = True
    else:
        synthetic = False

    try:
        profile_stages(wav_path)
    finally:
        if synthetic:
            os.unlink(wav_path)


if __name__ == "__main__":
    main()
