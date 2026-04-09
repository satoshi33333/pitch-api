# Pitch Analysis API

汎用音声ピッチ・リズム解析サーバー。ブラウザからの音声ファイルを受け取り、ピッチ（音程）・テンポ・音節情報を返します。

## エンドポイント

### GET /health
サーバー生存確認。

### POST /analyze
音声ファイルを解析してピッチ・リズム・音節情報を返す。

**パラメータ（multipart/form-data）:**
| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| file | File | 必須 | 音声ファイル（webm, wav, mp3, ogg） |
| min_pitch_hz | float | 60.0 | 検出する最低ピッチ（Hz） |
| max_pitch_hz | float | 800.0 | 検出する最高ピッチ（Hz） |
| confidence_threshold | float | 0.6 | ピッチ信頼度の閾値 |

**レスポンス例:**
```json
{
  "duration": 2.4,
  "tempo": 92.0,
  "rms_mean": 0.08,
  "brightness": 0.42,
  "syllable_count": 4,
  "syllables": [
    {"onset": 0.1, "duration": 0.3},
    {"onset": 0.5, "duration": 0.25}
  ],
  "pitch_count": 12,
  "pitches": [
    {"time": 0.12, "freq": 293.7, "midi": 62, "confidence": 0.91},
    {"time": 0.18, "freq": 311.1, "midi": 63, "confidence": 0.85}
  ]
}
```

## ローカル起動

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

ブラウザで `http://localhost:8000/docs` を開くとSwagger UIで試せます。

## Renderへのデプロイ

1. このリポジトリをGitHubにpush
2. [Render](https://render.com) でNew → Web Service
3. GitHubリポジトリを接続
4. 設定：
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Deploy

## フロントエンドからの使用例

```javascript
const formData = new FormData();
formData.append('file', audioBlob, 'recording.webm');

const res = await fetch('https://your-server.onrender.com/analyze', {
  method: 'POST',
  body: formData,
});
const data = await res.json();

// MIDIノート列を取得
const midiNotes = data.pitches.map(p => p.midi);
console.log('検出されたピッチ:', midiNotes);
```
