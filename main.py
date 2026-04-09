"""
Pitch Analysis + Claude Backing API v3.1
Python 3.14対応（soundfileで直接読み込み）
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import io
import httpx
from typing import Optional, List

app = FastAPI(title="Pitch Analysis API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


@app.get("/health")
def health():
    return {"status": "ok", "version": "3.1.0"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()

    # ffmpegでwebm→wavに変換してからsoundfileで読む
    import subprocess, shutil
    y, sr = None, None

    if shutil.which("ffmpeg"):
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
            tmp_in.write(raw)
            tmp_in_path = tmp_in.name
        tmp_out_path = tmp_in_path.replace(".webm", ".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "22050", "-ac", "1", tmp_out_path],
                capture_output=True, check=True
            )
            y, sr = sf.read(tmp_out_path, dtype="float32")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"ffmpeg変換失敗: {str(e)}")
        finally:
            os.unlink(tmp_in_path)
            if os.path.exists(tmp_out_path):
                os.unlink(tmp_out_path)
    else:
        # ffmpegがない場合はsoundfileで直接試みる
        try:
            y, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != 22050:
                y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                sr = 22050
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"読み込み失敗: {str(e)}")

    duration = float(len(y) / sr)
    hop_length = 256

    # pyin: 声専用高精度ピッチ検出
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    pitch_list = []
    for t, freq, voiced, prob in zip(times, f0, voiced_flag, voiced_prob):
        if not voiced or np.isnan(freq) or freq <= 0 or prob < 0.5:
            continue
        midi = int(round(librosa.hz_to_midi(freq)))
        midi = max(36, min(84, midi))
        pitch_list.append({
            "time": round(float(t), 4),
            "freq": round(float(freq), 2),
            "midi": midi,
            "confidence": round(float(prob), 3),
        })

    # テンポ
    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo = float(tempo_arr) if np.isscalar(tempo_arr) else float(tempo_arr[0])
    tempo = max(40.0, min(220.0, tempo))

    # 音節オンセット
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, backtrack=True, units="frames"
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    syllables = []
    for i, onset in enumerate(onset_times):
        end = float(onset_times[i + 1]) if i + 1 < len(onset_times) else duration
        syllables.append({"onset": round(float(onset), 4), "duration": round(end - float(onset), 4)})

    rms_mean = float(np.mean(librosa.feature.rms(y=y, hop_length=hop_length)[0]))
    brightness = min(1.0, float(np.mean(librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0])) * 50)

    # 音節ごとの代表ピッチ
    representative_midis = []
    for syl in syllables[:8]:
        start_t, end_t = syl["onset"], syl["onset"] + syl["duration"]
        midis_in_syl = [p["midi"] for p in pitch_list if start_t <= p["time"] <= end_t]
        if midis_in_syl:
            representative_midis.append(int(np.median(midis_in_syl)))

    return {
        "duration": round(duration, 3),
        "tempo": round(tempo, 1),
        "rms_mean": round(rms_mean, 4),
        "brightness": round(brightness, 3),
        "syllable_count": len(syllables),
        "syllables": syllables,
        "pitch_count": len(pitch_list),
        "pitches": pitch_list,
        "representative_midis": representative_midis,
    }


class ClaudeRequest(BaseModel):
    prompt: str
    history: Optional[List[dict]] = []


@app.post("/claude")
async def claude_proxy(req: ClaudeRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 400,
                "system": "You are a music composer. Respond with ONLY valid JSON, no markdown, no explanation.",
                "messages": [{"role": "user", "content": req.prompt}],
            }
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    import json
    text = (resp.json().get("content", [{}])[0].get("text", "")).strip()
    try:
        return json.loads(text)
    except Exception:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {text}")
