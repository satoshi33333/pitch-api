"""
Pitch Analysis + Claude Backing API v3
音声ピッチ解析＋Claude APIプロキシ
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import librosa
import tempfile
import os
import httpx
from typing import Optional, List

app = FastAPI(title="Pitch Analysis API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """音声ファイルを解析してピッチ・リズム・音節情報を返す（pyin使用）"""
    suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=422, detail=f"読み込み失敗: {str(e)}")
    finally:
        os.unlink(tmp_path)

    duration = float(librosa.get_duration(y=y, sr=sr))
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
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    syllables = []
    for i, onset in enumerate(onset_times):
        end = float(onset_times[i + 1]) if i + 1 < len(onset_times) else duration
        syllables.append({"onset": round(float(onset), 4), "duration": round(end - float(onset), 4)})

    rms_mean = float(np.mean(librosa.feature.rms(y=y, hop_length=hop_length)[0]))
    brightness = min(1.0, float(np.mean(librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0])) * 50)

    # 音節ごとの代表ピッチ（中央値）
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
    """Claude APIのプロキシ（CORSを回避するためサーバー経由で呼ぶ）"""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    messages = [{"role": "user", "content": req.prompt}]

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
                "messages": messages,
            }
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    text = (data.get("content", [{}])[0].get("text", "")).strip()
    # JSONとして返す
    import json
    try:
        return json.loads(text)
    except Exception:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {text}")
