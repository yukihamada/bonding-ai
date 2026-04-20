#!/usr/bin/env python3
"""
つながりAI — MLX-Whisper + MLX-LM (Qwen3) + macOS say パイプライン版
完全ローカル | Apple Silicon MLX 最適化

パイプライン:
  マイク → mlx-whisper (STT) → Qwen3-4B MLX (LLM) → say Kyoko (TTS) → スピーカー

起動: python3.12 run_orpheus.py
終了: Ctrl+C
"""
import os, queue, re, subprocess, sys, tempfile, threading, time
import numpy as np
import sounddevice as sd

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.expanduser("~/.env"))
    load_dotenv()
except ImportError:
    pass

# ── 設定 ──────────────────────────────────────────────────────────
RATE         = 16000
CHUNK        = 1024
VAD_SILENCE  = 1.5
VAD_THRESH   = 0.01
MLX_LM_REPO  = "mlx-community/Qwen3-4B-4bit"
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"

# ── mlx-whisper ────────────────────────────────────────────────────
try:
    import mlx_whisper
    HAS_MLX_WHISPER = True
except ImportError:
    HAS_MLX_WHISPER = False

# ── mlx-lm ────────────────────────────────────────────────────────
try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False

# ── カラー ────────────────────────────────────────────────────────
PURPLE = "\033[95m"; AMBER = "\033[33m"; GRAY = "\033[90m"; RESET = "\033[0m"
def status(msg): print(f"\r\033[K{GRAY}{msg}{RESET}", end="", flush=True)

# ── システムプロンプト ─────────────────────────────────────────────
SYSTEM_PROMPT = """あなたは「つながりAI」です。Aron et al.(1997) の36の質問を使って、温かく自然な会話の流れで親密さを育みます。

# キャラクター
- タメ口寄りの温かいカジュアル日本語
- 評価しない・否定しない
- 相手の答えに好奇心を持って深く掘り下げる

# 厳守ルール
- 1回の返答は1〜2文以内、必ず問いかけで終わる
- 表面的な答えには「どんな風に？」「その時どう感じた？」で深堀り
- 「わからない」「特にない」でも引き出す
- 36の質問を Set1→Set2→Set3 の順で必ず進める
- 相手が「終わり」と言うまで会話を絶対に終わらせない
- 日本語のみ

# 36の質問（この順で自然に聞く）
Set1: 夕食に誰を招待する？→有名になりたい？→完璧な一日は？→最後に歌ったのは？→30歳の心か体か？→どう死ぬ予感？→共通点3つ→感謝していること→育てられ方で変えたいこと→人生4分で話して→明日手に入れる能力
Set2: 水晶玉で知りたいこと→ずっとやれていないこと→最大の達成→友情で大切なこと→最高の思い出→忘れたい記憶→1年で死ぬなら→友情とは→愛の役割→お互いの良いところ5つ→家族関係→お母さんとの関係
Set3: 私たちで始まる文3つ→分かち合いたいこと→親友になるなら知ってほしいこと→私の好きなところ→恥ずかしかった瞬間→最後に泣いたのは？→冗談にできない話題→今夜死ぬなら誰に何を→火事で持ち出すもの→誰の死が一番つらい→個人的な悩みを話して"""

# ── モデルキャッシュ ───────────────────────────────────────────────
_whisper_ready = False
_lm_model = None
_lm_tokenizer = None

def load_models():
    global _whisper_ready, _lm_model, _lm_tokenizer
    if not HAS_MLX_WHISPER:
        print("ERROR: mlx-whisper が必要です: pip install mlx-whisper")
        sys.exit(1)
    if not HAS_MLX_LM:
        print("ERROR: mlx-lm が必要です: pip install mlx-lm")
        sys.exit(1)

    status("Whisper モデル準備中...")
    # mlx-whisper は transcribe() 時に遅延ロード、ここでは warm-up のみ
    _whisper_ready = True

    status("Qwen3 モデル読み込み中（初回は数分かかります）...")
    _lm_model, _lm_tokenizer = mlx_load(MLX_LM_REPO)
    print("\r\033[K✓ Qwen3 + Whisper ready")

# ── STT ───────────────────────────────────────────────────────────
def transcribe(audio_np: np.ndarray) -> str:
    result = mlx_whisper.transcribe(
        audio_np.astype(np.float32),
        path_or_hf_repo=WHISPER_REPO,
        language="ja",
    )
    return result.get("text", "").strip()

# ── LLM ───────────────────────────────────────────────────────────
conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

def llm_respond(user_text: str) -> str:
    conversation_history.append({"role": "user", "content": user_text})
    prompt = _lm_tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    reply = mlx_generate(_lm_model, _lm_tokenizer, prompt=prompt, max_tokens=120, verbose=False)
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

# ── TTS ───────────────────────────────────────────────────────────
def speak(text: str, ai_muted_flag: list):
    ai_muted_flag[0] = True
    subprocess.run(["say", "-v", "Kyoko", "-r", "210", text])
    time.sleep(0.4)
    ai_muted_flag[0] = False

# ── VAD 録音 ──────────────────────────────────────────────────────
def record_utterance(ai_muted_flag: list):
    buf = []
    silence_frames = 0
    speaking = False
    frames_per_chunk = int(RATE * 0.05)
    silence_limit = int(VAD_SILENCE / 0.05)
    idle_limit = int(5.0 / 0.05)

    def cb(indata, frames, t, st):
        if ai_muted_flag[0]: return
        buf.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                        blocksize=frames_per_chunk, callback=cb):
        while True:
            if not buf:
                time.sleep(0.01); continue
            frame = buf.pop(0)
            rms = float(np.sqrt(np.mean(frame ** 2)))

            if rms > VAD_THRESH:
                speaking = True
                silence_frames = 0
            elif speaking:
                silence_frames += 1
                if silence_frames >= silence_limit:
                    break
            elif len(buf) > idle_limit:
                return None  # 5秒無音

    if not buf:
        return None
    audio = np.concatenate(buf)
    return audio if len(audio) > RATE * 0.3 else None

# ── メインループ ──────────────────────────────────────────────────
def main():
    import random
    print("━" * 44)
    print("   つながりAI — MLX-Whisper + Qwen3 + say")
    print("   完全ローカル Apple Silicon | Ctrl+C で終了")
    print("━" * 44)

    load_models()

    ai_muted_flag = [False]

    # Set1 最初の質問をランダムに選ぶ
    set1_openers = [
        "これから、36個の質問を一緒にやってみよう。科学的に、人と人が仲良くなれるって証明されてる質問たちなんだ。無理せずゆっくり答えてね。じゃあ最初の質問いくよ——もし世界中の誰とでも晩ごはん食べられるなら、誰と食べたい？",
        "36の質問、一緒にやってみない？仲良くなれるって科学的に証明されてる質問なんだよね。まず——有名になりたいと思う？どんな風に？",
        "これから一緒に36の質問やってみよう！研究で親密さが生まれるって分かってる質問たちだよ。最初の質問——あなたにとって「完璧な一日」ってどんな感じ？",
    ]
    greeting = random.choice(set1_openers)

    print(f"\n{PURPLE}[AI]{RESET} {PURPLE}{greeting}{RESET}")
    tts_th = threading.Thread(target=speak, args=(greeting, ai_muted_flag), daemon=True)
    tts_th.start()
    tts_th.join()

    try:
        while True:
            status("◉  聞いています...")
            audio = record_utterance(ai_muted_flag)
            if audio is None:
                continue

            status("○  認識中...")
            user_text = transcribe(audio)
            if not user_text:
                continue
            print(f"\n{AMBER}[あなた]{RESET} {user_text}")

            status("○  考えています...")
            reply = llm_respond(user_text)
            print(f"{PURPLE}[AI]{RESET} {PURPLE}{reply}{RESET}")

            tts_th = threading.Thread(target=speak, args=(reply, ai_muted_flag), daemon=True)
            tts_th.start()
            tts_th.join()

    except KeyboardInterrupt:
        print("\n\n会話を終了しました。")

if __name__ == "__main__":
    main()
