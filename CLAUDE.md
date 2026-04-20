# つながりAI — bonding-ai

Aron et al.(1997) の36の質問を使った親密さ育成AIボイスエージェント。

## アーキテクチャ

```
ブラウザ (マイク/スピーカー)
  ↕ WebSocket (ws://localhost:8765/ws)
app.py (aiohttp サーバー)
  ├─ Pipeline モード: mlx-whisper → Qwen3.5 MLX-LM → edge-tts NanamiNeural
  └─ Moshi モード: moshi_mlx (S2S) WebSocket ブリッジ
                   ws://localhost:8998/api/chat
```

## 主要ファイル

| ファイル | 役割 |
|---------|------|
| `app.py` | メイン Webアプリ (aiohttp + WebSocket + HTML組み込み) |
| `run_orpheus.py` | ターミナル版パイプライン (Whisper+Qwen3+say Kyoko) |
| `run_moshi.py` | Moshi スタンドアロン起動 |
| `test_pipeline.py` | 自動品質テスト (TTS→Whisper CER + LLM応答チェック) |

## 起動方法

### app.py (Web UI) — メイン

```bash
# m5 Mac (192.168.0.5) で実行
ssh yukihamada@192.168.0.5
cd ~/bonding-ai
# 古いプロセスを確実に終了
lsof -ti:8765 | xargs kill -9 2>/dev/null; lsof -ti:8998 | xargs kill -9 2>/dev/null
python3.12 app.py
```

ブラウザで `http://192.168.0.5:8765` を開く。

### ローカルからアクセス（リモート起動）

```bash
# ローカルMacから直接起動してブラウザを開く
ssh yukihamada@192.168.0.5 "cd ~/bonding-ai && lsof -ti:8765 | xargs kill -9 2>/dev/null; nohup python3.12 app.py > /tmp/bonding.log 2>&1 &"
open http://192.168.0.5:8765
```

### ターミナル版 Pipeline (run_orpheus.py)

```bash
ssh yukihamada@192.168.0.5 "cd ~/bonding-ai && python3.12 run_orpheus.py"
# マイク → mlx-whisper → Qwen3-4B → say Kyoko
```

### Moshi スタンドアロン (run_moshi.py)

```bash
ssh yukihamada@192.168.0.5 "cd ~/bonding-ai && python3.12 run_moshi.py"
# Moshi Web UI: http://192.168.0.5:8998
```

## 設定値 (app.py)

| 設定 | 値 | 説明 |
|-----|-----|------|
| `PORT` | 8765 | Web UIポート |
| `MLX_LM_REPO` | `mlx-community/Qwen3.5-122B-A10B-4bit` | LLM (Pipeline) |
| `WHISPER_REPO` | `mlx-community/whisper-large-v3` | STT |
| `TTS_VOICE` | `ja-JP-NanamiNeural` | edge-tts音声 |
| `MOSHI_REPO` | `kyutai/moshika-mlx-q4` | Moshi 日本語版 |
| `MODE` | `"moshi"` | デフォルトモード |
| `MOSHI_RATE` | 24000 | Moshi サンプルレート |
| `VAD_THRESH` | 0.015 | 音声検出閾値 (RMS) |
| `VAD_SILENCE` | 1.5 | 無音判定秒数 |

## モデルの場所 (m5 Mac)

- Whisper: HuggingFace cache (`~/.cache/huggingface/hub/`)
- Qwen3.5-122B: MLX cache (`~/.cache/huggingface/hub/`)
- Moshika: HuggingFace cache

## Moshi モード技術詳細

⚠️ **重要: Moshi は現状英語のみ** — `kyutai/moshika-mlx-q4` / `kyutai/moshiko-mlx-q4` どちらも英語モデル（moshika=女性声、moshiko=男性声）。MLX対応の日本語Moshiモデルは2026-04時点で未リリース。
→ **デフォルトは Pipeline モード**。Moshiタブは「英語実験モード」として残す。

- `moshi_mlx.local_web` で WebSocket サーバー (port 8998) 起動
- プロトコル: `b"\x00"` ハンドシェイク → クライアント `b"\x01"+opus` → サーバー `b"\x01"+opus` + `b"\x02"+text`
- Opus codec: `sphn.OpusStreamWriter/Reader` (24kHz)
- ブラウザ: 16kHz PCM → resample → 24kHz Opus → Moshi → 24kHz Opus → 24kHz Float32 PCM → AudioContext

### Moshi動作確認方法
```bash
ssh yukihamada@192.168.0.5 "cd ~/bonding-ai && /opt/homebrew/bin/python3.12 test_moshi.py 2>&1 | tee /tmp/moshi_test.log"
```
- AI挨拶受信 / 音声受信 / 応答テキスト受信を自動検証
- 日本語入力に対し英語で応答するのは仕様（Moshi英語モデルのため）

## トラブルシューティング

### ポートが占有されている

```bash
lsof -ti:8765 | xargs kill -9
lsof -ti:8998 | xargs kill -9
```

### Moshi が英語を話す

`MOSHI_REPO = "kyutai/moshika-mlx-q4"` を確認（moshiko ではなく moshika）

### ollama が Metal でクラッシュ (M5 Max)

M5 Max では ggml Metal shader がクラッシュする。必ず `mlx-lm` を使用。

### モデルロードが遅い

Qwen3.5-122B は初回で数分かかる。`/tmp/bonding.log` でステータス確認。

## 自動テスト

```bash
ssh yukihamada@192.168.0.5 "cd ~/bonding-ai && python3.12 test_pipeline.py 2>&1 | tee /tmp/pipeline_test.log"
```

- Phase 1: TTS→Whisper CER (目標 < 15%)
- Phase 2: LLM応答品質 (問いかけ終わり・長さ・日本語)
- Phase 3: フルラウンドトリップ会話

## テスト→改善ループ（Claude Codeが自律実行する）

### 手順
1. `test_pipeline.py` を m5 で実行してテスト結果を取得
2. Phase 1 (STT) CER > 15% → Whisper `initial_prompt` を調整
3. Phase 2 (LLM) 合格率 < 80% → SYSTEM_PROMPT を強化（問いかけルール・ユーモア）
4. Phase 3 (会話継続) 合格率 < 80% → 多ターン応答の整合性を確認
5. **全項目パスしたらユーザーに報告**（「テスト完了：プロダクション品質です」）

### 合格基準
- STT平均CER < 15%
- LLM応答合格率 ≥ 80%（問いかけ終わり・日本語・適切な長さ・過剰な丁寧語なし）
- 会話継続合格率 ≥ 80%

### テスト実行コマンド
```bash
ssh yukihamada@192.168.0.5 "cd ~/bonding-ai && /opt/homebrew/bin/python3.12 test_pipeline.py 2>&1 | tee /tmp/pipeline_test.log && echo DONE"
```

### プロセス管理
```bash
# app.py 再起動
ssh yukihamada@192.168.0.5 "pkill -9 -f 'bonding-ai/app.py' 2>/dev/null; sleep 1; cd ~/bonding-ai && nohup /opt/homebrew/bin/python3.12 app.py > /tmp/bonding.log 2>&1 &"
# ログ確認
ssh yukihamada@192.168.0.5 "tail -f /tmp/bonding.log"
```

## A/Bテスト（挨拶バリアント自動評価）

8種類の挨拶戦略（A〜H）をランダムに出し分け、エンゲージメントを計測。

### 結果確認
```bash
curl http://192.168.0.5:8765/api/ab_results
```

### ログ場所
`~/bonding-ai/conversations/ab_log.jsonl`

### 評価指標（Opus設計）
| 指標 | 説明 | 目標 |
|------|------|------|
| avg_chars_per_turn | 1ターンあたりの発話文字数（多い=より深く話している） | 30文字以上 |
| avg_turns | 平均会話ターン数 | 5ターン以上 |
| avg_duration_sec | 平均セッション時間 | 720秒（12分）以上 |

バリアントごとに **最低30セッション**集まったら優位差判定可能。  
勝者が決まったら `GREETING_VARIANTS` から負けバリアントを削除し、勝者を増やす。

## 挨拶バリアント戦略一覧（Opus設計 v2 — 思わず答えてしまう15パターン）

人間が答えずにいられない心理フックを使用（ネガティビティ・バイアス、最近記憶、二択、反論誘発、共感トリガー等）。

| ID | 戦略 | 心理フック |
|----|------|-----------|
| I1 | ネガティビティ | 嫌いな食べ物は即答できる |
| I2 | 最近記憶 | 今日のイラッ（小さな不満） |
| I3 | 二択 | 朝型/夜型（認知負荷最小） |
| I4 | パターン補完 | もし1週間休みなら |
| I5 | 反論誘発 | 朝活いい説、本当？ |
| I6 | 具体エピソード | 1週間以内の笑い |
| I7 | 地味な自慢 | これだけは自信ある |
| I8 | 自己認識 | めんどくさい性格？ |
| I9 | 数値評価 | 今日10点満点で何点？ |
| I10 | 共感ムカつき | コンビニで詰められる |
| I11 | あるある共感 | 日曜夜の憂鬱 |
| I12 | 最近記憶（買物） | 直近買ってよかった |
| I13 | 二択（性格） | 旅行、計画派？ノリ派？ |
| I14 | 連想・匂い | 子供の頃の夏休み |
| I15 | 愚痴誘発 | 最近愚痴りたいこと |

**Opus予測のトップ3:** I1 > I11 > I6

## 依存ライブラリ

```
aiohttp, edge-tts, mlx-whisper, mlx-lm, sounddevice, numpy, scipy, sphn, websockets
```

インストール: `pip install aiohttp edge-tts mlx-whisper mlx-lm sounddevice numpy scipy sphn websockets`
