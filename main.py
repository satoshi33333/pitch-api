"""
Pitch Analysis API
汎用音声ピッチ解析サーバー
- POST /analyze  : 音声ファイル → ピッチ・リズム・音節情報
- GET  /health   : サーバー生存確認
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import tempfile
import os
from typing import Optional

app = FastAPI(
    title="Pitch Analysis API",
    description="汎用音声ピッチ・リズム解析API",
    version="1.0.0"
)

# CORS設定（ブラウザからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    min_pitch_hz: float = 60.0,    # 検出する最低ピッチ（Hz）
    max_pitch_hz: float = 800.0,   # 検出する最高ピッチ（Hz）
    hop_length: int = 512,         # 分析フレームのホップ長
    confidence_threshold: float = 0.6,  # ピッチ信頼度の閾値
):
    """
    音声ファイルを解析してピッチ・リズム・音節情報を返す。

    Parameters:
    - file: 音声ファイル（webm, wav, mp3, ogg など）
    - min_pitch_hz: 検出する最低ピッチ（デフォルト: 60Hz）
    - max_pitch_hz: 検出する最高ピッチ（デフォルト: 800Hz）
    - confidence_threshold: ピッチ信頼度の閾値（デフォルト: 0.6）

    Returns:
    - pitches: 時刻ごとのピッチ情報（time, freq, midi, confidence）
    - tempo: 推定BPM
    - syllables: 音節オンセット情報（onset時刻, duration）
    - duration: 音声の長さ（秒）
    - rms_mean: 平均音量（エネルギー）
    - brightness: 明るさ（ゼロ交差率ベース）
    """

    # 音声ファイルを一時ファイルに保存
    suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # librosaで音声読み込み（モノラル、22050Hz）
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=422, detail=f"音声ファイルの読み込みに失敗: {str(e)}")
    finally:
        os.unlink(tmp_path)

    duration = float(librosa.get_duration(y=y, sr=sr))

    # =====================
    # 1. ピッチ検出（piptrack）
    # =====================
    pitches_raw, magnitudes = librosa.piptrack(
        y=y, sr=sr,
        hop_length=hop_length,
        fmin=min_pitch_hz,
        fmax=max_pitch_hz,
    )

    pitch_list = []
    times = librosa.frames_to_time(
        np.arange(pitches_raw.shape[1]), sr=sr, hop_length=hop_length
    )

    for t_idx, t in enumerate(times):
        # 各フレームで最大マグニチュードのピッチを取得
        idx = magnitudes[:, t_idx].argmax()
        freq = float(pitches_raw[idx, t_idx])
        mag = float(magnitudes[idx, t_idx])

        if freq < min_pitch_hz or freq > max_pitch_hz:
            continue

        # マグニチュードを正規化して信頼度として使用
        max_mag = float(magnitudes[:, t_idx].max()) if magnitudes[:, t_idx].max() > 0 else 1.0
        confidence = mag / max_mag if max_mag > 0 else 0.0

        if confidence < confidence_threshold:
            continue

        # 周波数 → MIDIノート番号
        midi = int(round(librosa.hz_to_midi(freq)))
        midi = max(0, min(127, midi))

        pitch_list.append({
            "time": round(float(t), 4),
            "freq": round(freq, 2),
            "midi": midi,
            "confidence": round(confidence, 3),
        })

    # =====================
    # 2. テンポ・ビート検出
    # =====================
    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo = float(tempo_arr) if np.isscalar(tempo_arr) else float(tempo_arr[0])
    tempo = max(40.0, min(220.0, tempo))

    # =====================
    # 3. 音節オンセット検出
    # =====================
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units="frames",
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    syllables = []
    for i, onset in enumerate(onset_times):
        end = float(onset_times[i + 1]) if i + 1 < len(onset_times) else duration
        syllables.append({
            "onset": round(float(onset), 4),
            "duration": round(end - float(onset), 4),
        })

    # =====================
    # 4. 全体的な音声特徴
    # =====================
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    zcr_mean = float(np.mean(zcr))
    brightness = min(1.0, zcr_mean * 50)

    return {
        "duration": round(duration, 3),
        "tempo": round(tempo, 1),
        "rms_mean": round(rms_mean, 4),
        "brightness": round(brightness, 3),
        "syllable_count": len(syllables),
        "syllables": syllables,
        "pitch_count": len(pitch_list),
        "pitches": pitch_list,
    }
