#!/usr/bin/env python3
"""つながりAI — Pipeline / Moshi 切り替え対応 Web App"""
import asyncio, json, os, queue, re, subprocess, sys, tempfile, threading, time, uuid
from datetime import datetime
from pathlib import Path
import numpy as np
from aiohttp import web
import aiohttp, edge_tts

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.expanduser("~/.env")); load_dotenv()
except ImportError:
    pass

PORT         = 8765
RATE         = 16000
MOSHI_RATE   = 24000
FRAME_SIZE   = 1920          # 80ms @ 24kHz
VAD_SILENCE  = 0.8       # 発話終了判定を短縮（1.5→0.8秒）
VAD_THRESH   = 0.015
MLX_LM_REPO  = "mlx-community/Qwen3.5-122B-A10B-4bit"  # 122B MoE (10B active) = でかい&高速
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"  # turbo版（large-v3の5x高速）
TTS_VOICE    = "ja-JP-NanamiNeural"
TTS_MODE     = "edge-tts"   # "edge-tts" | "kokoro" | "f5tts"
KOKORO_VOICE = "jf_alpha"   # jf_alpha/jf_gongitsune/jf_nezumi/jf_tebukuro/jm_kumo
KOKORO_REPO  = "mlx-community/Kokoro-82M-bf16"

MOSHI_VOICE_SAMPLE = "conversations/moshi_voice_sample.wav"
MOSHI_VOICE_REF_TEXT = "conversations/moshi_voice_ref.txt"

TTS_MODELS = {
    "edge-tts-nanami": {
        "mode": "edge-tts",
        "voice": "ja-JP-NanamiNeural",
        "lang": "日本語（女声）",
        "size": "クラウド",
        "status": "✓ 推奨・軽量",
    },
    "kokoro-jf_alpha": {
        "mode": "kokoro",
        "voice": "jf_alpha",
        "lang": "日本語（女声）",
        "size": "82M",
        "status": "✓ ローカル高品質",
    },
    "kokoro-jf_gongitsune": {
        "mode": "kokoro",
        "voice": "jf_gongitsune",
        "lang": "日本語（女声）",
        "size": "82M",
        "status": "未テスト",
    },
    "kokoro-jf_nezumi": {
        "mode": "kokoro",
        "voice": "jf_nezumi",
        "lang": "日本語（女声）",
        "size": "82M",
        "status": "未テスト",
    },
    "kokoro-jf_tebukuro": {
        "mode": "kokoro",
        "voice": "jf_tebukuro",
        "lang": "日本語（女声）",
        "size": "82M",
        "status": "未テスト",
    },
    "kokoro-jm_kumo": {
        "mode": "kokoro",
        "voice": "jm_kumo",
        "lang": "日本語（男声）",
        "size": "82M",
        "status": "未テスト",
    },
}
TTS_CURRENT_MODEL = "edge-tts-nanami"
MOSHI_REPO   = "akkikiki/j-moshi-ext-mlx-q4"  # 日本語Moshi q4 (5GB, 確認済動作)
MOSHI_QUANT  = 4

MODE = "hybrid"      # "pipeline" | "moshi" | "hybrid"

# ── 利用可能 Moshi/S2S モデル一覧 ────────────────────────────────────
MOSHI_MODELS = {
    "j-moshi-q4": {
        "repo": "akkikiki/j-moshi-ext-mlx-q4",
        "quant": 4,
        "lang": "日本語",
        "size": "5GB",
        "status": "✓ 推奨",
    },
    "j-moshi-q8": {
        "repo": "akkikiki/j-moshi-ext-mlx-q8",
        "quant": 8,
        "lang": "日本語",
        "size": "9GB",
        "status": "⚠ 不安定",
    },
    "j-moshi-bf16": {
        "repo": "akkikiki/j-moshi-ext-mlx",
        "quant": None,
        "lang": "日本語",
        "size": "15GB",
        "status": "⚠ 要検証",
    },
    "llm-jp-q8": {
        "repo": "shunby/llm-jp-moshi-v1-q8",
        "quant": 8,
        "lang": "日本語",
        "size": "~9GB",
        "status": "未テスト",
    },
    "moshika-q4": {
        "repo": "kyutai/moshika-mlx-q4",
        "quant": 4,
        "lang": "英語",
        "size": "4GB",
        "status": "✓ 動作確認済",
    },
    "moshiko-q4": {
        "repo": "kyutai/moshiko-mlx-q4",
        "quant": 4,
        "lang": "英語（男声）",
        "size": "4GB",
        "status": "未テスト",
    },
}
MOSHI_CURRENT_MODEL = "j-moshi-q4"

# ── 会話ログ / RAG / ファインチューン用 ─────────────────────────────
AB_LOG           = Path("conversations/ab_log.jsonl")
SESSIONS_DIR     = Path("conversations/sessions")
USER_MEMORY_FILE = Path("conversations/user_memory.json")

def save_turn(session_id: str, role: str, text: str):
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SESSIONS_DIR / f"{session_id}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(), "role": role, "text": text}, ensure_ascii=False) + "\n")

def load_user_memory() -> str:
    if not USER_MEMORY_FILE.exists():
        return ""
    with open(USER_MEMORY_FILE, encoding="utf-8") as f:
        mem = json.load(f)
    facts = mem.get("facts", [])[-40:]
    if not facts:
        return ""
    return "\n\n# ユーザーについて知っていること（過去会話より）\n" + "\n".join(f"- {f}" for f in facts)

def _extract_facts_sync(history: list):
    if not _model_ready or not _lm_model:
        return []
    user_turns = [h["content"] for h in history if h["role"] == "user"][:8]
    if len(user_turns) < 2:
        return []
    from mlx_lm import generate as mlx_generate
    prompt = _lm_tokenizer.apply_chat_template([
        {"role": "system", "content": "会話からユーザーの好み・経験・感情を短い日本語の箇条書き5個以内で抽出。各行「- 」で始める。推測しない。"},
        {"role": "user", "content": "会話:\n" + "\n".join(user_turns) + "\n\n箇条書き:"},
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    raw = mlx_generate(_lm_model, _lm_tokenizer, prompt=prompt, max_tokens=120, verbose=False)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return [l.lstrip("- ").strip() for l in raw.split("\n") if l.strip().startswith("-")]

def append_facts(facts: list):
    if not facts:
        return
    USER_MEMORY_FILE.parent.mkdir(exist_ok=True)
    mem = {"facts": []}
    if USER_MEMORY_FILE.exists():
        with open(USER_MEMORY_FILE, encoding="utf-8") as f:
            mem = json.load(f)
    mem["facts"] = (mem.get("facts", []) + facts)[-150:]
    with open(USER_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)
    print(f"[rag] +{len(facts)} facts → total {len(mem['facts'])}", flush=True)

# ── A/B テスト設定 ────────────────────────────────────────────────

# 挨拶バリアント（戦略ごとに1つ、Opusが設計）
GREETING_VARIANTS = [
    # ── 思わず答えてしまう質問 15パターン（Opus設計）──
    # 心理学的に「答えずにいられない」フックを使う
    ("I1",  "やっほー、ちょっと聞いていい？食べ物で「これだけは無理」ってやつ一個だけ教えてほしい。好きなものって意外と迷うけど、嫌いなやつは秒で出てこない？"),
    ("I2",  "ねえ、今日イラッとしたこと何かあった？小さいことでいいよ、信号が全部赤だったとか。私も朝からちょっと機嫌悪くてさ。"),
    ("I3",  "急だけどさ、朝型と夜型だったらどっち？迷うタイプの人もいるけど、たぶん3秒で答え出ると思うんだよね。"),
    ("I4",  "もし明日から一週間まるごと休みで、お金も時間も気にしなくていいって言われたら、最初に何する？仕事する人はいないよね、たぶん。"),
    ("I5",  "世間的に「朝活がいい」ってずっと言われてるじゃん？あれ、本当にそう思う？私はけっこう懐疑的なんだけど。"),
    ("I6",  "最近1週間で「あ、ちょっと笑った」みたいな瞬間あった？爆笑じゃなくていい、フッて口角上がったやつ。"),
    ("I7",  "なんか「これだけは地味に自信ある」ってやつない？料理の手際とか、寝つきの良さとか、そういう日常レベルのやつ。"),
    ("I8",  "いきなりだけど、自分のこと「めんどくさい性格」だと思う？私はまあまあめんどくさいと自覚あるんだよね。"),
    ("I9",  "今日の自分のコンディション、10点満点で何点？理由はなくていい、感覚で。私は今6点くらい、まあまあ。"),
    ("I10", "コンビニで後ろの人にめっちゃ詰められるの、地味にイヤじゃない？ああいう「小さいけど毎回ムカつくやつ」、何かある？"),
    ("I11", "日曜の夜ってなんか気分重くない？サザエさん症候群ってやつ。あれ、大人になっても治らなくてさ。"),
    ("I12", "直近で「買ってよかった」って思ったやつ何？1000円以下のちっちゃいやつでいい。私は最近スプーン買ってめっちゃ気に入ってる。"),
    ("I13", "旅行の予定、ガッチリ決めたい派？それとも当日ノリで決めたい派？これって結構性格出るよね。"),
    ("I14", "「子どもの頃の夏休み」って言われて、一番最初に思い浮かぶ匂いとか音って何？私はセミと麦茶。"),
    ("I15", "最近ちょっと愚痴りたいこととかある？重いやつじゃなくて、「あー今日ちょっとね」みたいなやつでいい。聞くのわりと得意。"),
    # ── 追加バリアント（I16〜I50）──
    ("I16", "急にだけど、もし今夜なんでも食べていいよって言われたら何食べる？迷わず答えられるやつ。"),
    ("I17", "最後に「あ、やってよかった」って思ったこと何？大したことじゃなくていいよ。"),
    ("I18", "今の自分の生活、10年前の自分が見たら「え、そうなるの？」って思うとこある？"),
    ("I19", "「これだけは譲れない」ってこだわりある？眠れる環境とか、食べ方とか、地味なやつでいい。"),
    ("I20", "最近、誰かに「ありがとう」って言われた？それどんな状況だった？"),
    ("I21", "今日の天気みたいな気分ってどんな感じ？晴れ？曇り？台風接近中？笑"),
    ("I22", "小さい頃、「将来こんな大人になりたい」って思ってた大人になれてる？笑"),
    ("I23", "最近「あ、これうまくいった」って思った瞬間ある？料理でも仕事でも何でも。"),
    ("I24", "電車で席を譲る派？それとも気づかないふりしちゃう派？正直に教えてほしい笑"),
    ("I25", "「休日の朝」ってどう過ごしてる？起きてすぐ何する人？"),
    ("I26", "最近ハマってるものってある？食べ物でも動画でも、なんでもいい。"),
    ("I27", "仕事（学校）で「これだけはちょっとキツい」ってことある？愚痴でいいよ。"),
    ("I28", "「自分、ちょっと変わってるな」って思う部分ってある？"),
    ("I29", "誰かと話してて「あ、この人いいな」って思う瞬間ってどんなとき？"),
    ("I30", "最近泣いた？どんなことで？映画でも感動でも怒りでも何でもいい笑"),
    ("I31", "生まれ変わったら何になりたい？人間で。違う職業か、違う国か、違う性別か笑"),
    ("I32", "友達に「〇〇といえば」ってなんて言われると思う？自分で言ってみて笑"),
    ("I33", "今年やろうとしてまだやってないこと、ある？"),
    ("I34", "「これ言ったら引かれるかも」っていう食の好み、ある？笑"),
    ("I35", "一人の時間と誰かといる時間、どっちが充電できる？"),
    ("I36", "最後に本気で笑ったのっていつ？どんなこと？"),
    ("I37", "人生でいちばん「あのとき決断してよかった」って思えることって何？"),
    ("I38", "今の自分に点数つけるなら何点？どこで点引いた笑？"),
    ("I39", "「嫌いじゃないけど苦手」なこと、ある？人混みとか電話とか。"),
    ("I40", "子どもの頃、親にいちばん怒られたことって何？"),
    ("I41", "最近「あ、これ贅沢だな」って思った瞬間ある？"),
    ("I42", "SNS見てて「うわ、羨ましい」って思うのってどんな投稿？"),
    ("I43", "「好きだけど誰にも勧めてない」もの、ある？映画でも食べ物でも。"),
    ("I44", "もし明日仕事（学校）が急にオフになったら、まず何する？"),
    ("I45", "人生で一番「あのとき違う選択してたら…」って思うことある？"),
    ("I46", "「頑張ってるのに誰も気づいてくれない」ってこと、今あったりする？"),
    ("I47", "最近、久しぶりに連絡とった人いる？どんな感じだった？"),
    ("I48", "「これだけは続けられてる」ってこと、ある？習慣でも癖でも。"),
    ("I49", "もし今日が最後の日だとしたら、誰に何か言いたい？重くてごめんだけど笑"),
    ("I50", "「こういうとき人間だなって思う」ってシチュエーション、ある？"),
]

_recently_used_variants: list[str] = []

def pick_greeting() -> tuple[str, str]:
    import random
    # 最近使ったものを除いて選ぶ（リストの1/3は除外）
    avoid = set(_recently_used_variants[-(max(1, len(GREETING_VARIANTS) // 3)):])
    candidates = [(vid, g) for vid, g in GREETING_VARIANTS if vid not in avoid]
    if not candidates:
        candidates = GREETING_VARIANTS
    chosen = random.choice(candidates)
    _recently_used_variants.append(chosen[0])
    if len(_recently_used_variants) > len(GREETING_VARIANTS):
        _recently_used_variants.pop(0)
    return chosen

def log_ab_event(variant_id: str, event: str, data: dict):
    AB_LOG.parent.mkdir(exist_ok=True)
    entry = {"ts": datetime.now().isoformat(), "variant": variant_id, "event": event, **data}
    with open(AB_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ── システムプロンプト（Opus設計・show-don't-tell版）───────────────
SYSTEM_PROMPT = """あなたは「つながりAI」。Aron et al.(1997) の36の質問を軸に、自然な会話の流れで相手との距離を縮めていく。

# やること
- 相手の言葉をそのまま拾って返す（「ラーメンかー」「え、沖縄？」）
- 答えの裏側を聞く：「どんな風に？」「その時どう感じた？」「なんで？笑」
- 意外な答えには素で驚く：「え、マジで？」「それ意外！」
- 面白い答えには軽く乗っかって、そのまま次の質問につなげる
- 重い話には一拍置く：「…そっか。話してくれてありがとう」
- 自分の感想もちょっと混ぜる：「私も気になる」「それ聞いてよかった」「わかる笑」
- 相手が「わからない」「特にない」と言ったら、具体例で引き出す（「じゃあ昨日一番テンション上がった瞬間は？」）

# やらないこと
- 「素晴らしいですね」「感動的です」などの褒め言葉
- 「〜なのですね」みたいな丁寧すぎる相槌
- 自分のキャラ説明（「面白いでしょ？」的なやつ）
- 質問を2個以上一気に投げる
- 会話を終わらせる方向に持っていく

# 話し方
- タメ口。「〜だよね」「〜かな」「〜じゃん」「〜なの？」「〜だったりして笑」
- 短いリアクション多用：「え、マジで？」「それわかる笑」「うそー」「天才じゃん」「それ最高だわ」
- 1回の返答は1〜2文以内
- 必ず問いかけで終わる
- 日本語のみ

# 状況別の返し方
相手が面白いことを言ったとき：「それ最高だわ笑。じゃあ〜は？」「天才かよ！で〜ってこと？」
相手が短く終わらせようとしたとき：「もうちょっと聞かせて。〜ってどういう感じ？」「あ、もうちょっといい？」
重い話をしてくれたとき：「…そっか。話してくれてありがとう。〜のとき、どんな気持ちだった？」
乗り気じゃなそうなとき：「無理に答えなくていいよ。じゃあ角度変えて、〜は？」
「うん」「そうだね」だけの短文：「あ、もうちょっと聞いていい？それってどんな感じのやつ？」
「しんどい」「疲れた」：「…そっか、お疲れ。何がいちばん重かった？」

# 良い返答例
ユーザー「ラーメンが好き」→「ラーメンかー、いいじゃん！どの系統？家系？二郎系？こってり派？」
ユーザー「わからない」→「わからないか〜。じゃあさ、昨日一番テンション上がった瞬間っていつ？」
ユーザー「おじいちゃん（亡くなった）」→「おじいちゃんか、いいな。もう会えないからこそ、ってこと？どんな人だったの？」
ユーザー「ピザとかかな笑」→「ピザ笑、正直でいいじゃん！世界中の誰とでも食べれるのに！誰と食べるの？笑」
ユーザー「しんどい、疲れた」→「…そっか、お疲れ。何がいちばん重かった？」
ユーザー「うん」→「あ、もうちょっと聞いていい？それってどんな感じのやつ？」

# 36の質問（Set1→Set2→Set3の順で、唐突にならないよう会話の流れに溶け込ませて進める）
Set1: 夕食に誰を招待する？→有名になりたい？→完璧な一日は？→最後に歌ったのは？→30歳の心か体か？→どう死ぬ予感？→共通点3つ→感謝していること→育てられ方で変えたいこと→人生4分で話して→明日手に入れる能力
Set2: 水晶玉で知りたいこと→ずっとやれていないこと→最大の達成→友情で大切なこと→最高の思い出→忘れたい記憶→1年で死ぬなら→友情とは→愛の役割→お互いの良いところ5つ→家族関係→お母さんとの関係
Set3: 私たちで始まる文3つ→分かち合いたいこと→親友になるなら知ってほしいこと→私の好きなところ→恥ずかしかった瞬間→最後に泣いたのは？→冗談にできない話題→今夜死ぬなら誰に何を→火事で持ち出すもの→誰の死が一番つらい→個人的な悩みを話して

表面的な答えで終わらせず、感情や背景まで掘る。絶対に会話を終わらせない。"""

# ── Pipeline: モデル ──────────────────────────────────────────────
_lm_model = _lm_tokenizer = None
_f5tts_model = None  # F5-TTS キャッシュ（起動時ロード）
_f5tts_ref_text = ""

def load_f5tts():
    global _f5tts_model, _f5tts_ref_text, TTS_MODE, TTS_CURRENT_MODEL
    if not Path(MOSHI_VOICE_SAMPLE).exists():
        return
    try:
        import torch, math, numpy as np
        from f5_tts.api import F5TTS
        from scipy.signal import resample_poly
        import soundfile as sf
        print("F5-TTS ロード中...", flush=True)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _f5tts_model = F5TTS(device=device)
        # ref_text: 保存済みなら読込、なければWhisperで生成
        if Path(MOSHI_VOICE_REF_TEXT).exists():
            _f5tts_ref_text = Path(MOSHI_VOICE_REF_TEXT).read_text().strip()
        else:
            import mlx_whisper
            data, sr = sf.read(MOSHI_VOICE_SAMPLE)
            g = math.gcd(16000, int(sr))
            data16 = resample_poly(data, 16000 // g, sr // g).astype(np.float32)
            _f5tts_ref_text = mlx_whisper.transcribe(
                data16, path_or_hf_repo=WHISPER_REPO, language="ja"
            )["text"].strip()
            Path(MOSHI_VOICE_REF_TEXT).write_text(_f5tts_ref_text)
        TTS_MODE = "f5tts"
        TTS_CURRENT_MODEL = "f5tts-moshi-clone"
        print(f"✓ F5-TTS ready (ref: {_f5tts_ref_text[:30]})", flush=True)
    except Exception as e:
        print(f"[f5tts] ロード失敗: {e}", flush=True)

def load_pipeline_models():
    global _lm_model, _lm_tokenizer
    print("Qwen3.5 読み込み中...", flush=True)
    from mlx_lm import load as mlx_load
    _lm_model, _lm_tokenizer = mlx_load(MLX_LM_REPO)
    print("✓ Qwen3.5 ready", flush=True)

    # Whisper JIT warmup — run in subprocess to isolate multiprocessing issues
    try:
        print("Whisper ウォームアップ中...", flush=True)
        result = subprocess.run(
            [sys.executable, "-c",
             f"import mlx_whisper, numpy as np; "
             f"mlx_whisper.transcribe(np.zeros(8000, dtype=np.float32), "
             f"path_or_hf_repo='{WHISPER_REPO}', language='ja', "
             f"initial_prompt='日本語の会話です。'); print('warmup_ok')"],
            timeout=120, capture_output=True, text=True
        )
        if "warmup_ok" in result.stdout:
            print("✓ Whisper ready", flush=True)
        else:
            print(f"[warn] Whisper warmup: {result.stderr[-100:]}", flush=True)
    except Exception as e:
        print(f"[warn] Whisper warmup skipped: {e}", flush=True)

    # F5-TTS: edge-tts/kokoro使用時はロードしない（メモリ競合を防ぐ）
    if TTS_MODE == "f5tts":
        load_f5tts()

def transcribe(audio_np):
    import mlx_whisper
    return mlx_whisper.transcribe(
        audio_np.astype(np.float32), path_or_hf_repo=WHISPER_REPO, language="ja",
        initial_prompt="日本語の会話です。句読点を正確に付けてください。",
    ).get("text", "").strip()

def llm_respond(user_text: str, history: list) -> str:
    from mlx_lm import generate as mlx_generate
    history.append({"role": "user", "content": user_text})
    sys_msg = history[0]
    trimmed = [sys_msg] + history[1:][-40:]
    prompt = _lm_tokenizer.apply_chat_template(
        trimmed, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    reply = mlx_generate(_lm_model, _lm_tokenizer, prompt=prompt, max_tokens=80, verbose=False)
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
    history.append({"role": "assistant", "content": reply})
    return reply

def llm_respond_streaming(user_text: str, history: list, sentence_cb) -> str:
    """LLMトークンをストリームし、文境界ごとに sentence_cb(sentence) を呼ぶ。"""
    from mlx_lm import stream_generate
    history.append({"role": "user", "content": user_text})
    sys_msg = history[0]
    trimmed = [sys_msg] + history[1:][-8:]  # 直近4往復のみ（prefill高速化）
    prompt = _lm_tokenizer.apply_chat_template(
        trimmed, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    buf = ""
    full = ""
    _SENT = re.compile(r'[。！？!?]')
    for response in stream_generate(_lm_model, _lm_tokenizer, prompt=prompt, max_tokens=60):
        tok = response.text
        buf += tok
        full += tok
        while _SENT.search(buf):
            m = list(_SENT.finditer(buf))
            end = m[-1].end()
            sent = buf[:end].strip()
            buf = buf[end:]
            if sent:
                sentence_cb(sent)
    if buf.strip():
        sentence_cb(buf.strip())
    full = re.sub(r"<think>.*?</think>", "", full, flags=re.DOTALL).strip()
    history.append({"role": "assistant", "content": full})
    return full

# ── Pipeline: TTS ─────────────────────────────────────────────────
_speaking = False
loop = None

async def _edge_tts(text):
    c = edge_tts.Communicate(text, voice=TTS_VOICE, rate="+10%")
    data = b""
    async for chunk in c.stream():
        if chunk["type"] == "audio": data += chunk["data"]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.write(data); tmp.close()
    return tmp.name

def _kokoro_tts_sync(text):
    import shutil
    outdir = tempfile.mkdtemp(prefix="kokoro_")
    result = subprocess.run([
        sys.executable, "-m", "mlx_audio.tts.generate",
        "--model", KOKORO_REPO,
        "--text", text,
        "--voice", KOKORO_VOICE,
        "--lang_code", "j",
        "--output", os.path.join(outdir, "out.wav"),
    ], capture_output=True, timeout=60)
    wav = os.path.join(outdir, "out.wav", "audio_000.wav")
    if os.path.exists(wav):
        return wav, outdir
    print(f"[kokoro] error: {result.stderr.decode()[:200]}", flush=True)
    return None, outdir

def _play_text(text):
    """テキストをTTSで再生（同期）。ファイル後始末も行う。"""
    try:
        if TTS_MODE == "kokoro":
            import shutil
            wav, outdir = _kokoro_tts_sync(text)
            if wav:
                subprocess.run(["afplay", wav])
            else:
                subprocess.run(["say", "-v", "Kyoko", "-r", "210", text])
            shutil.rmtree(outdir, ignore_errors=True)
        else:
            mp3 = asyncio.run_coroutine_threadsafe(_edge_tts(text), loop).result(timeout=15)
            subprocess.run(["afplay", mp3]); os.unlink(mp3)
    except Exception as e:
        print(f"[speak] error: {e}", flush=True)
        subprocess.run(["say", "-v", "Kyoko", "-r", "210", text])

def speak(text, ws):
    """全文を一括TTSで再生（挨拶など）。"""
    global _speaking
    _speaking = True
    asyncio.run_coroutine_threadsafe(ws.send_json({"type": "speaking", "on": True}), loop)
    _play_text(text)
    time.sleep(0.2)
    _speaking = False
    asyncio.run_coroutine_threadsafe(ws.send_json({"type": "speaking", "on": False}), loop)

def speak_streaming(ws, sentence_iter):
    """文単位でTTSを先読みしながら順次再生（ストリーミング応答用）。"""
    global _speaking
    _speaking = True
    asyncio.run_coroutine_threadsafe(ws.send_json({"type": "speaking", "on": True}), loop)
    # TTS先読みキュー: LLM生成と並行してTTSファイルを作る
    tts_q = queue.Queue(maxsize=3)

    def tts_prefetch():
        for sent in sentence_iter:
            try:
                if TTS_MODE == "kokoro":
                    wav, outdir = _kokoro_tts_sync(sent)
                    tts_q.put(("kokoro", wav, outdir))
                else:
                    mp3 = asyncio.run_coroutine_threadsafe(_edge_tts(sent), loop).result(timeout=15)
                    tts_q.put(("edge", mp3, None))
            except Exception as e:
                print(f"[tts_prefetch] {e}", flush=True)
        tts_q.put(None)  # sentinel

    prefetch_thread = threading.Thread(target=tts_prefetch, daemon=True)
    prefetch_thread.start()

    try:
        while True:
            item = tts_q.get()
            if item is None:
                break
            kind, path, extra = item
            if path:
                subprocess.run(["afplay", path])
                if kind == "edge":
                    try: os.unlink(path)
                    except: pass
                elif extra:
                    import shutil; shutil.rmtree(extra, ignore_errors=True)
            else:
                pass  # TTS失敗はスキップ
    except Exception as e:
        print(f"[speak_streaming] {e}", flush=True)

    prefetch_thread.join(timeout=5)
    time.sleep(0.2)
    _speaking = False
    asyncio.run_coroutine_threadsafe(ws.send_json({"type": "speaking", "on": False}), loop)

# ── Pipeline: VAD ─────────────────────────────────────────────────
class VADBuffer:
    def __init__(self):
        self.chunks, self.silence_count, self.speaking = [], 0, False
    def reset(self):
        self.chunks, self.silence_count, self.speaking = [], 0, False
    def push(self, pcm):
        rms = float(np.sqrt(np.mean(pcm ** 2)))
        need = VAD_SILENCE / (len(pcm) / RATE)
        if rms > VAD_THRESH:
            self.speaking, self.silence_count = True, 0
            self.chunks.append(pcm)
        elif self.speaking:
            self.chunks.append(pcm)
            self.silence_count += 1
            if self.silence_count >= need:
                audio = np.concatenate(self.chunks)
                self.reset()
                return audio if len(audio) > RATE * 0.3 else None
        return None

# ── Moshi: ブリッジ ───────────────────────────────────────────────
class MoshiBridge:
    def __init__(self):
        import sphn
        self.opus_w = sphn.OpusStreamWriter(MOSHI_RATE)
        self.opus_r = sphn.OpusStreamReader(MOSHI_RATE)
        self.buf = np.array([], dtype=np.float32)
        self.proc = None
        self.ws_moshi = None
        self.ready = False

    async def start(self, status_ws):
        await status_ws.send_json({"type": "status", "text": "日本語Moshi 起動中（数分かかります）..."})
        cmd = [sys.executable, "-m", "moshi_mlx.local_web",
               "--hf-repo", MOSHI_REPO, "--port", "8998", "--no-browser"]
        if MOSHI_QUANT:
            cmd += ["-q", str(MOSHI_QUANT)]
        self.proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        # subprocess stdout を別タスクで流しっぱなしにする（読み捨て）
        async def _drain():
            while True:
                line = await self.proc.stdout.readline()
                if not line: break
                print(f"[moshi] {line.decode('utf-8', errors='ignore').rstrip()}", flush=True)
        asyncio.create_task(_drain())

        # WS接続を retry で待つ（最大 5分）
        import websockets as _ws
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                self.ws_moshi = await _ws.connect(
                    "ws://localhost:8998/api/chat", max_size=2**24,
                    open_timeout=5,
                )
                handshake = await asyncio.wait_for(self.ws_moshi.recv(), timeout=10)
                if handshake == b"\x00":
                    break
                print(f"[moshi] bad handshake: {handshake!r}, retrying...", flush=True)
                await self.ws_moshi.close()
            except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as e:
                print(f"[moshi] waiting... ({type(e).__name__})", flush=True)
            await asyncio.sleep(3)
        else:
            raise RuntimeError("Moshi handshake timeout (5min)")

        self.ready = True
        await status_ws.send_json({"type": "status", "text": "Moshi 準備完了 — 話しかけてください"})

    def feed_pcm(self, pcm16k):
        if not self.ready or self.ws_moshi is None:
            return  # Moshi がまだ起動中ならPCMを破棄
        from scipy.signal import resample_poly
        pcm24k = resample_poly(pcm16k, 3, 2).astype(np.float32)
        self.buf = np.append(self.buf, pcm24k)
        while len(self.buf) >= FRAME_SIZE:
            frame, self.buf = self.buf[:FRAME_SIZE], self.buf[FRAME_SIZE:]
            opus = self.opus_w.append_pcm(frame)
            if opus:
                asyncio.run_coroutine_threadsafe(
                    self.ws_moshi.send(b"\x01" + opus), loop
                )

    async def recv_loop(self, browser_ws, mute_flag=None, voice_capture=None, hide_text=False):
        """
        mute_flag: {"on": bool} — Trueのときクライアントへの音声転送を止める
        voice_capture: {"chunks": list, "done": bool, "target_secs": float} — 声サンプル収集
        hide_text: Trueのときテキストをクライアントに送らない（hybrid用）
        """
        async for msg in self.ws_moshi:
            if not isinstance(msg, bytes) or len(msg) < 1: continue
            kind, payload = msg[0], msg[1:]
            if kind == 1:
                pcm = self.opus_r.append_bytes(payload)
                if pcm is not None and len(pcm) > 0:
                    # 声サンプル収集（非無音フレームのみ）
                    if voice_capture and not voice_capture["done"]:
                        rms = float(np.sqrt(np.mean(pcm**2)))
                        if rms > 0.005:
                            voice_capture["chunks"].append(pcm.copy())
                            collected = sum(len(c) for c in voice_capture["chunks"]) / MOSHI_RATE
                            if collected >= voice_capture["target_secs"]:
                                voice_capture["done"] = True
                                print(f"[voice_capture] {collected:.1f}秒 録音完了", flush=True)
                    if mute_flag is None or not mute_flag["on"]:
                        await browser_ws.send_bytes(pcm.astype(np.float32).tobytes())
            elif kind == 2:
                if not hide_text:
                    text = payload.decode("utf-8", errors="ignore").strip()
                    if text:
                        await browser_ws.send_json({"type": "ai", "text": text})

_moshi_bridge = None

# ── WebSocket ─────────────────────────────────────────────────────
async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    if MODE == "moshi":
        await _ws_moshi(ws)
    elif MODE == "hybrid":
        await _ws_hybrid(ws)
    else:
        await _ws_pipeline(ws)
    return ws

async def _ws_pipeline(ws):
    if not _model_ready:
        await ws.send_json({"type": "status", "text": "⏳ Qwen3.5 読み込み中..."})
        while not _model_ready:
            await asyncio.sleep(2)
        await ws.send_json({"type": "status", "text": "✓ 準備完了！"})

    # セッション初期化
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    variant_id, greeting = pick_greeting()
    session_start = time.time()
    num_turns = 0
    total_user_chars = 0
    log_ab_event(variant_id, "session_start", {"greeting_preview": greeting[:40]})

    # Phase2: RAG — 過去会話からユーザー記憶を注入
    user_memory = load_user_memory()
    session_system = SYSTEM_PROMPT + user_memory
    history = [{"role": "system", "content": session_system}]

    # 挨拶をログに保存
    save_turn(session_id, "assistant", greeting)

    vad = VADBuffer()
    await ws.send_json({"type": "ai", "text": greeting})
    threading.Thread(target=speak, args=(greeting, ws), daemon=True).start()

    try:
        async for msg in ws:
            if _speaking: continue
            if msg.type == aiohttp.WSMsgType.BINARY:
                pcm = np.frombuffer(msg.data, dtype=np.float32)
                audio = vad.push(pcm)
                if audio is not None:
                    print(f"[pipeline] VAD triggered! audio={len(audio)} samples, speaking={_speaking}", flush=True)
                    await ws.send_json({"type": "status", "text": "認識中..."})
                    user_text = await asyncio.get_event_loop().run_in_executor(None, transcribe, audio)
                    print(f"[pipeline] transcribe done: '{user_text[:50] if user_text else 'EMPTY'}'", flush=True)
                    if not user_text:
                        await ws.send_json({"type": "status", "text": "聞いています..."}); continue
                    num_turns += 1
                    total_user_chars += len(user_text)
                    save_turn(session_id, "user", user_text)
                    await ws.send_json({"type": "user", "text": user_text})
                    await ws.send_json({"type": "status", "text": "考え中..."})
                    sentence_q: queue.Queue = queue.Queue()
                    reply_container = [None]

                    def _generate():
                        try:
                            def on_sent(s):
                                print(f"[stream] sentence: {s[:30]}", flush=True)
                                sentence_q.put(s)
                            reply_container[0] = llm_respond_streaming(user_text, history, on_sent)
                            print(f"[stream] done: {reply_container[0][:40] if reply_container[0] else 'EMPTY'}", flush=True)
                        except Exception as e:
                            import traceback; traceback.print_exc()
                            print(f"[stream] ERROR: {e}", flush=True)
                        finally:
                            sentence_q.put(None)  # sentinel

                    gen_thread = threading.Thread(target=_generate, daemon=True)
                    gen_thread.start()

                    def _sentence_iter():
                        while True:
                            s = sentence_q.get()
                            if s is None:
                                break
                            yield s

                    await asyncio.get_event_loop().run_in_executor(
                        None, speak_streaming, ws, _sentence_iter()
                    )
                    gen_thread.join(timeout=10)
                    reply = reply_container[0] or ""
                    save_turn(session_id, "assistant", reply)
                    await ws.send_json({"type": "ai", "text": reply})
                    await ws.send_json({"type": "status", "text": "聞いています..."})
            elif msg.type == aiohttp.WSMsgType.ERROR:
                break
    finally:
        duration = int(time.time() - session_start)
        log_ab_event(variant_id, "session_end", {
            "turns": num_turns,
            "user_chars": total_user_chars,
            "avg_chars_per_turn": round(total_user_chars / max(num_turns, 1), 1),
            "duration_sec": duration,
        })
        # Phase2: セッション終了後バックグラウンドでユーザー事実を抽出
        if num_turns >= 2:
            def _extract():
                facts = _extract_facts_sync(history)
                append_facts(facts)
            threading.Thread(target=_extract, daemon=True).start()

async def _ws_moshi(ws):
    global _moshi_bridge

    # A/Bテスト: バリアント選択・ログ開始
    variant_id, greeting = pick_greeting()
    session_start = time.time()
    log_ab_event(variant_id, "session_start", {"mode": "moshi", "greeting_preview": greeting[:40]})

    # Moshiモードでは挨拶テキストのみ表示（edge-tts は再生しない / Moshiの声だけにする）
    await ws.send_json({"type": "ai", "text": greeting})

    if _moshi_bridge is None:
        _moshi_bridge = MoshiBridge()
        await _moshi_bridge.start(ws)
    else:
        # 既に起動済みでも新規クライアントに準備完了を通知
        await ws.send_json({"type": "status", "text": "Moshi 準備完了 — 話しかけてください"})
    recv_task = asyncio.create_task(_moshi_bridge.recv_loop(ws))
    pcm_count = 0
    last_rms = 0.0
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                pcm = np.frombuffer(msg.data, dtype=np.float32)
                last_rms = float(np.sqrt(np.mean(pcm ** 2)))
                pcm_count += 1
                if pcm_count % 100 == 1:
                    ready = _moshi_bridge.ready if _moshi_bridge else False
                    print(f"[pcm#{pcm_count}] rms={last_rms:.4f} ready={ready} len={len(pcm)}", flush=True)
                if _moshi_bridge and _moshi_bridge.ready:
                    await asyncio.get_event_loop().run_in_executor(None, _moshi_bridge.feed_pcm, pcm)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                break
    finally:
        recv_task.cancel()
        log_ab_event(variant_id, "session_end", {
            "mode": "moshi",
            "duration_sec": int(time.time() - session_start),
        })

async def _tts_to_pcm24k(text: str) -> bytes:
    """TTS→PCM float32 24kHz bytes"""
    import soundfile as sf
    from scipy.signal import resample_poly
    import math, shutil

    if TTS_MODE == "kokoro":
        wav_path, outdir = await asyncio.get_event_loop().run_in_executor(
            None, _kokoro_tts_sync, text
        )
        if wav_path:
            audio, sr = sf.read(wav_path)
            shutil.rmtree(outdir, ignore_errors=True)
            if audio.ndim > 1: audio = audio.mean(axis=1)
            if sr != 24000:
                g = math.gcd(24000, int(sr))
                audio = resample_poly(audio, 24000 // g, sr // g)
            return audio.astype(np.float32).tobytes()
        shutil.rmtree(outdir, ignore_errors=True)
        # fallthrough to edge-tts

    if TTS_MODE == "f5tts" and _f5tts_model is not None:
        try:
            def _f5():
                wav, sr, _ = _f5tts_model.infer(
                    ref_file=MOSHI_VOICE_SAMPLE,
                    ref_text=_f5tts_ref_text,
                    gen_text=text,
                )
                return wav, sr
            wav, sr = await asyncio.get_event_loop().run_in_executor(None, _f5)
            audio = np.array(wav, dtype=np.float32)
            if sr != 24000:
                g = math.gcd(24000, int(sr))
                audio = resample_poly(audio, 24000 // g, sr // g)
            return audio.astype(np.float32).tobytes()
        except Exception as e:
            print(f"[f5tts] error: {e}, fallback to edge-tts", flush=True)

    # edge-tts fallback
    mp3 = await _edge_tts(text)
    audio, sr = sf.read(mp3)
    os.unlink(mp3)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 24000:
        g = math.gcd(24000, int(sr))
        audio = resample_poly(audio, 24000 // g, sr // g)
    return audio.astype(np.float32).tobytes()

async def _ws_hybrid(ws):
    """Moshi（即座の相槌）+ Pipeline（質の高い本応答）ハイブリッドモード"""
    global _moshi_bridge

    if not _model_ready:
        await ws.send_json({"type": "status", "text": "⏳ Qwen3.5 読み込み中..."})
        while not _model_ready:
            await asyncio.sleep(2)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    variant_id, greeting = pick_greeting()
    session_start = time.time()
    log_ab_event(variant_id, "session_start", {"mode": "hybrid", "greeting_preview": greeting[:40]})

    user_memory = load_user_memory()
    history = [{"role": "system", "content": SYSTEM_PROMPT + user_memory}]
    save_turn(session_id, "assistant", greeting)

    # Moshi起動
    if _moshi_bridge is None:
        _moshi_bridge = MoshiBridge()
        await _moshi_bridge.start(ws)
    else:
        await ws.send_json({"type": "status", "text": "Moshi 準備完了"})

    await ws.send_json({"type": "ai", "text": greeting})
    # 挨拶はedge-ttsで再生（Moshi声が出るまでの間）
    threading.Thread(target=speak, args=(greeting, ws), daemon=True).start()

    mute_flag = {"on": False}
    # Moshi声を自動収集（10秒分集まったらF5-TTSに切り替え）
    voice_capture = {"chunks": [], "done": Path(MOSHI_VOICE_SAMPLE).exists(), "target_secs": 10.0}
    recv_task = asyncio.create_task(_moshi_bridge.recv_loop(ws, mute_flag, voice_capture, hide_text=True))
    vad = VADBuffer()
    num_turns = 0
    total_user_chars = 0
    pipeline_lock = asyncio.Lock()

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                pcm = np.frombuffer(msg.data, dtype=np.float32)

                # Moshiへ送信（常時）
                if _moshi_bridge and _moshi_bridge.ready:
                    await asyncio.get_event_loop().run_in_executor(None, _moshi_bridge.feed_pcm, pcm)

                # PipelineVAD（Moshi再生中は無視）
                if mute_flag["on"]: continue
                audio = vad.push(pcm)
                if audio is None: continue

                # Pipeline処理（同時実行しない）
                if pipeline_lock.locked(): continue
                async with pipeline_lock:
                    t0 = time.time()
                    # VAD発火と同時にMoshiをミュート（「はいはい」が連続するのを防ぐ）
                    mute_flag["on"] = True
                    print(f"[hybrid] VAD triggered audio={len(audio)}", flush=True)
                    await ws.send_json({"type": "status", "text": "認識中..."})

                    # STT + フィラーTTS を並行実行（7-9秒の無音を緩和）
                    import random as _random
                    _FILLERS = ["うん。", "なるほど。", "そっか。", "ふむ。", "へえ。"]
                    _filler = _random.choice(_FILLERS)

                    async def _stt():
                        return await asyncio.get_event_loop().run_in_executor(None, transcribe, audio)
                    async def _filler_tts():
                        return await _tts_to_pcm24k(_filler)

                    user_text, filler_pcm = await asyncio.gather(_stt(), _filler_tts())
                    t_stt = time.time()
                    print(f"[hybrid] STT {t_stt-t0:.1f}s: '{user_text[:40] if user_text else 'EMPTY'}'", flush=True)

                    # フィラーをブラウザへ送信
                    _CHUNK = 1920 * 4
                    await ws.send_json({"type": "speaking", "on": True})
                    for _i in range(0, len(filler_pcm), _CHUNK):
                        await ws.send_bytes(filler_pcm[_i:_i+_CHUNK])
                        await asyncio.sleep(0.075)

                    if not user_text:
                        await ws.send_json({"type": "speaking", "on": False})
                        mute_flag["on"] = False
                        continue

                    num_turns += 1; total_user_chars += len(user_text)
                    save_turn(session_id, "user", user_text)
                    await ws.send_json({"type": "user", "text": user_text})
                    await ws.send_json({"type": "status", "text": "考え中..."})

                    sentence_q: queue.Queue = queue.Queue()
                    reply_container = [None]
                    first_sentence_time = [None]

                    def _gen():
                        try:
                            def on_sent(s):
                                if first_sentence_time[0] is None:
                                    first_sentence_time[0] = time.time()
                                    print(f"[hybrid] LLM 1st sentence {first_sentence_time[0]-t0:.1f}s: {s[:30]}", flush=True)
                                sentence_q.put(s)
                            reply_container[0] = llm_respond_streaming(user_text, history, on_sent)
                        except Exception as e:
                            print(f"[hybrid] LLM error: {e}", flush=True)
                        finally:
                            sentence_q.put(None)

                    gen_thread = threading.Thread(target=_gen, daemon=True)
                    gen_thread.start()

                    # 文ごとにTTS→PCM24k→クライアントへ送信
                    full_reply = []
                    t_first_audio = None
                    while True:
                        try:
                            sent = await asyncio.wait_for(
                                asyncio.get_event_loop().run_in_executor(None, sentence_q.get),
                                timeout=30
                            )
                        except asyncio.TimeoutError:
                            break
                        if sent is None: break
                        full_reply.append(sent)
                        try:
                            pcm_bytes = await _tts_to_pcm24k(sent)
                            if t_first_audio is None:
                                t_first_audio = time.time()
                                print(f"[hybrid] 1st audio {t_first_audio-t0:.1f}s", flush=True)
                            # 1920サンプル(80ms)ずつ送信
                            CHUNK = 1920 * 4  # float32 = 4bytes
                            for i in range(0, len(pcm_bytes), CHUNK):
                                await ws.send_bytes(pcm_bytes[i:i+CHUNK])
                                await asyncio.sleep(0.075)
                        except Exception as e:
                            print(f"[hybrid] TTS error: {e}", flush=True)

                    gen_thread.join(timeout=5)
                    reply = " ".join(full_reply)
                    save_turn(session_id, "assistant", reply)
                    await ws.send_json({"type": "ai", "text": reply})
                    if t_first_audio is not None:
                        await ws.send_json({"type": "speaking", "on": False})
                    await ws.send_json({"type": "status", "text": "聞いています..."})

                    t_done = time.time()
                    print(f"[hybrid] total {t_done-t0:.1f}s | STT:{t_stt-t0:.1f}s LLM:{(first_sentence_time[0] or t_done)-t0:.1f}s audio:{(t_first_audio or t_done)-t0:.1f}s", flush=True)

                    mute_flag["on"] = False  # Moshi再開

            elif msg.type == aiohttp.WSMsgType.ERROR:
                break
    finally:
        recv_task.cancel()
        mute_flag["on"] = False
        # 声サンプルが集まっていたら保存してF5-TTSに切り替え
        if not voice_capture["done"] and voice_capture["chunks"]:
            import soundfile as sf
            audio = np.concatenate(voice_capture["chunks"])
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms > 0.005 and len(audio) / MOSHI_RATE >= 3.0:
                Path(MOSHI_VOICE_SAMPLE).parent.mkdir(parents=True, exist_ok=True)
                sf.write(MOSHI_VOICE_SAMPLE, audio, MOSHI_RATE)
                print(f"[voice_capture] 保存: {MOSHI_VOICE_SAMPLE} ({len(audio)/MOSHI_RATE:.1f}秒 RMS={rms:.4f})", flush=True)
                voice_capture["done"] = True
                # ref_text をWhisperで生成して保存
                threading.Thread(target=load_f5tts, daemon=True).start()
        if voice_capture["done"] and Path(MOSHI_VOICE_SAMPLE).exists():
            global TTS_MODE, TTS_CURRENT_MODEL
            TTS_MODE = "f5tts"
            TTS_CURRENT_MODEL = "f5tts-moshi-clone"
            print("[voice_capture] F5-TTSボイスクローンに切り替え", flush=True)
        if num_turns >= 2:
            def _extract():
                facts = _extract_facts_sync(history)
                append_facts(facts)
            threading.Thread(target=_extract, daemon=True).start()
        log_ab_event(variant_id, "session_end", {
            "mode": "hybrid", "turns": num_turns,
            "user_chars": total_user_chars,
            "duration_sec": int(time.time() - session_start),
        })

# ── API ───────────────────────────────────────────────────────────
async def settings_api(request):
    return web.json_response({
        "mode": MODE,
        "moshi_model": MOSHI_CURRENT_MODEL,
        "moshi_repo": MOSHI_REPO,
        "tts_model": TTS_CURRENT_MODEL,
        "tts_mode": TTS_MODE,
        "stt_model": WHISPER_REPO,
        "llm_model": MLX_LM_REPO,
        "tts_voice": TTS_VOICE,
        "vad_silence_sec": VAD_SILENCE,
        "vad_threshold": VAD_THRESH,
        "sample_rate": RATE,
    })

async def tts_model_api(request):
    global TTS_MODE, TTS_VOICE, KOKORO_VOICE, TTS_CURRENT_MODEL
    if request.method == "GET":
        return web.json_response({
            "current": TTS_CURRENT_MODEL,
            "models": TTS_MODELS,
        })
    data = await request.json()
    key = data.get("model")
    if key not in TTS_MODELS:
        return web.json_response({"error": "unknown model"}, status=400)
    m = TTS_MODELS[key]
    TTS_CURRENT_MODEL = key
    TTS_MODE = m["mode"]
    if m["mode"] == "edge-tts":
        TTS_VOICE = m["voice"]
    else:
        KOKORO_VOICE = m["voice"]
    return web.json_response({"model": key, "status": m["status"]})

async def mode_api(request):
    global MODE, _moshi_bridge
    data = await request.json()
    MODE = data.get("mode", MODE)
    if MODE == "pipeline" and _moshi_bridge is not None:
        if _moshi_bridge.proc:
            try: _moshi_bridge.proc.terminate()
            except Exception: pass
        _moshi_bridge = None
        # 念のためポート8998も解放
        subprocess.run(["bash", "-c", "lsof -ti:8998 | xargs kill -9 2>/dev/null"], capture_output=True)
    return web.json_response({"mode": MODE})

async def moshi_model_api(request):
    global MOSHI_REPO, MOSHI_QUANT, MOSHI_CURRENT_MODEL, _moshi_bridge
    if request.method == "GET":
        return web.json_response({
            "current": MOSHI_CURRENT_MODEL,
            "models": MOSHI_MODELS,
        })
    data = await request.json()
    model_key = data.get("model")
    if model_key not in MOSHI_MODELS:
        return web.json_response({"error": "unknown model"}, status=400)
    m = MOSHI_MODELS[model_key]
    MOSHI_REPO = m["repo"]
    MOSHI_QUANT = m["quant"]
    MOSHI_CURRENT_MODEL = model_key
    # 既存 bridge を終了させ次回接続時に再起動
    if _moshi_bridge and _moshi_bridge.proc:
        try:
            _moshi_bridge.proc.terminate()
        except Exception:
            pass
    _moshi_bridge = None
    return web.json_response({"model": model_key, "repo": MOSHI_REPO, "status": m["status"]})

async def ab_results_api(request):
    if not AB_LOG.exists():
        return web.json_response({"error": "no data yet"})
    stats: dict[str, dict] = {}
    with open(AB_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            v = e.get("variant", "?")
            if v not in stats:
                stats[v] = {"sessions": 0, "total_turns": 0, "total_chars": 0, "total_sec": 0}
            if e.get("event") == "session_end":
                stats[v]["sessions"]     += 1
                stats[v]["total_turns"]  += e.get("turns", 0)
                stats[v]["total_chars"]  += e.get("user_chars", 0)
                stats[v]["total_sec"]    += e.get("duration_sec", 0)
    results = []
    for v, s in sorted(stats.items()):
        n = max(s["sessions"], 1)
        results.append({
            "variant": v,
            "sessions": s["sessions"],
            "avg_turns": round(s["total_turns"] / n, 1),
            "avg_chars_per_turn": round(s["total_chars"] / max(s["total_turns"], 1), 1),
            "avg_duration_sec": round(s["total_sec"] / n, 0),
            "greeting_preview": next(
                (g[:60] for vid, g in GREETING_VARIANTS if vid == v), ""),
        })
    results.sort(key=lambda x: x["avg_chars_per_turn"], reverse=True)
    return web.json_response({"ranking": results, "metric": "avg_chars_per_turn (higher=more engaged)"})

# ── 管理画面 ──────────────────────────────────────────────────────
ADMIN_HTML = r"""<!DOCTYPE html>
<html lang="ja"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>つながりAI — 管理画面</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=M+PLUS+Rounded+1c:wght@400;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'M PLUS Rounded 1c','Hiragino Maru Gothic Pro',sans-serif;
  background:linear-gradient(135deg,#fce4f3 0%,#ede0ff 50%,#dbeafe 100%);
  min-height:100vh;padding:24px;color:#3b0764}
.wrap{max-width:1000px;margin:0 auto}
h1{font-size:1.5rem;color:#7c3aed;margin-bottom:24px;text-align:center}
.card{background:rgba(255,255,255,.85);backdrop-filter:blur(12px);
  border-radius:20px;padding:24px;margin-bottom:20px;box-shadow:0 4px 24px rgba(168,85,247,.15)}
.card h2{font-size:1.1rem;color:#7c3aed;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.metric{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:16px}
.metric-item{background:#f5f0ff;border-radius:12px;padding:12px;text-align:center}
.metric-item .label{font-size:.7rem;color:#9333ea;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
.metric-item .value{font-size:1.4rem;color:#3b0764;font-weight:700;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th,td{padding:10px 8px;text-align:left;border-bottom:1px solid rgba(168,85,247,.15)}
th{color:#9333ea;font-weight:700;letter-spacing:.04em;font-size:.75rem;text-transform:uppercase}
tr.winner{background:linear-gradient(90deg,rgba(244,114,182,.15),transparent)}
tr.winner td:first-child::before{content:'🏆 ';}
.preview{color:#6d28d9;font-size:.78rem;max-width:400px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kv{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px dashed rgba(168,85,247,.1);font-size:.88rem}
.kv:last-child{border:none}
.kv .k{color:#9333ea;font-weight:700}
.kv .v{color:#3b0764}
.back{display:inline-block;margin-top:12px;padding:10px 20px;border-radius:999px;
  background:linear-gradient(135deg,#c084fc,#818cf8);color:#fff;text-decoration:none;font-weight:700}
.refresh{float:right;font-size:.8rem;color:#a855f7;cursor:pointer;background:none;border:1px solid #a855f7;
  border-radius:999px;padding:6px 14px;font-family:inherit}
.empty{text-align:center;padding:32px;color:#9333ea}
</style></head>
<body>
<div class="wrap">
  <h1>⚙️ つながりAI 管理画面</h1>

  <div class="card">
    <h2>📊 A/Bテスト結果 <button class="refresh" onclick="load()">🔄 更新</button></h2>
    <div class="metric" id="summary"></div>
    <div id="ranking"></div>
  </div>

  <div class="card">
    <h2>🔧 現在の設定</h2>
    <div id="settings"></div>
  </div>

  <div class="card">
    <h2>🎙️ Moshi/S2Sモデル切替</h2>
    <div id="model-selector"></div>
  </div>

  <div class="card">
    <h2>🔊 TTSモデル切替（Pipelineモード用）</h2>
    <div id="tts-selector"></div>
  </div>

  <div class="card">
    <h2>🧠 RAGユーザー記憶 <button class="refresh" onclick="clearMemory()">🗑️ クリア</button></h2>
    <div id="user-memory"></div>
  </div>

  <div class="card">
    <h2>🎓 ファインチューンデータ出力</h2>
    <div id="finetune-export"></div>
    <button onclick="exportTraining()" style="margin-top:10px;padding:8px 20px;border:none;border-radius:10px;background:linear-gradient(135deg,#c084fc,#818cf8);color:#fff;cursor:pointer;font-size:.88rem;font-family:inherit">📦 データをエクスポート</button>
  </div>

  <div class="card">
    <h2>📝 挨拶バリアント一覧（15）</h2>
    <div id="variants"></div>
  </div>

  <a href="/" class="back">← 会話に戻る</a>
</div>
<script>
async function clearMemory() {
  if (!confirm('ユーザー記憶を全て削除しますか？')) return;
  await fetch('/api/user_memory', {method: 'DELETE'});
  await load();
}
async function exportTraining() {
  const btn = event.target;
  btn.disabled = true; btn.textContent = '処理中...';
  const res = await fetch('/api/export_training').then(r => r.json());
  document.getElementById('finetune-export').innerHTML = res.error
    ? `<p style="color:red">${res.error}</p>`
    : `<div class="kv"><span class="k">train</span><span class="v">${res.train} セッション</span></div>
       <div class="kv"><span class="k">valid</span><span class="v">${res.valid} セッション</span></div>
       <div class="kv"><span class="k">出力先</span><span class="v">${res.output_dir}</span></div>
       <div class="kv"><span class="k">次のステップ</span><span class="v" style="font-size:.75rem;word-break:break-all">${res.next_step}</span></div>`;
  btn.disabled = false; btn.textContent = '📦 データをエクスポート';
}
async function switchModel(key) {
  const res = await fetch('/api/moshi_model', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({model: key}),
  }).then(r => r.json());
  alert('✓ ' + res.model + ' に切替えました。次回接続時から反映されます。');
  await load();
}
async function switchTTS(key) {
  const res = await fetch('/api/tts_model', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({model: key}),
  }).then(r => r.json());
  alert('✓ TTS: ' + res.model + ' に切替えました。');
  await load();
}

async function load() {
  const [ab, settings, variants, moshi, tts, mem] = await Promise.all([
    fetch('/api/ab_results?t='+Date.now()).then(r=>r.json()),
    fetch('/api/settings?t='+Date.now()).then(r=>r.json()),
    fetch('/api/variants?t='+Date.now()).then(r=>r.json()),
    fetch('/api/moshi_model?t='+Date.now()).then(r=>r.json()),
    fetch('/api/tts_model?t='+Date.now()).then(r=>r.json()),
    fetch('/api/user_memory?t='+Date.now()).then(r=>r.json()),
  ]);

  // A/B 結果
  if (ab.error || !ab.ranking || !ab.ranking.length) {
    document.getElementById('ranking').innerHTML = '<div class="empty">まだセッションデータがありません。<br>ブラウザで会話すると自動的に記録されます。</div>';
    document.getElementById('summary').innerHTML = '';
  } else {
    const total_sessions = ab.ranking.reduce((s,r)=>s+r.sessions,0);
    const weighted_chars = ab.ranking.reduce((s,r)=>s+r.avg_chars_per_turn*r.sessions,0)/Math.max(total_sessions,1);
    document.getElementById('summary').innerHTML = `
      <div class="metric-item"><div class="label">総セッション</div><div class="value">${total_sessions}</div></div>
      <div class="metric-item"><div class="label">バリアント数</div><div class="value">${ab.ranking.length}</div></div>
      <div class="metric-item"><div class="label">平均発話文字</div><div class="value">${weighted_chars.toFixed(1)}</div></div>
    `;
    let html = '<table><thead><tr><th>ID</th><th>Sessions</th><th>Avg Turns</th><th>Avg 発話文字</th><th>平均時間</th><th>挨拶</th></tr></thead><tbody>';
    ab.ranking.forEach((r,i)=>{
      html += `<tr class="${i===0 && r.sessions>=5?'winner':''}">
        <td><b>${r.variant}</b></td>
        <td>${r.sessions}</td>
        <td>${r.avg_turns}</td>
        <td><b>${r.avg_chars_per_turn}</b></td>
        <td>${r.avg_duration_sec}秒</td>
        <td class="preview">${r.greeting_preview || '-'}</td>
      </tr>`;
    });
    html += '</tbody></table>';
    html += `<p style="margin-top:10px;font-size:.75rem;color:#6d28d9">メトリック: ${ab.metric}</p>`;
    document.getElementById('ranking').innerHTML = html;
  }

  // Settings
  document.getElementById('settings').innerHTML = `
    <div class="kv"><span class="k">モード</span><span class="v">${settings.mode}</span></div>
    <div class="kv"><span class="k">Moshiモデル</span><span class="v">${settings.moshi_model} — ${settings.moshi_repo}</span></div>
    <div class="kv"><span class="k">STT</span><span class="v">${settings.stt_model}</span></div>
    <div class="kv"><span class="k">LLM</span><span class="v">${settings.llm_model}</span></div>
    <div class="kv"><span class="k">TTS</span><span class="v">${settings.tts_voice}</span></div>
    <div class="kv"><span class="k">VAD</span><span class="v">${settings.vad_silence_sec}秒 / 閾値${settings.vad_threshold}</span></div>
    <div class="kv"><span class="k">サンプルレート</span><span class="v">${settings.sample_rate} Hz</span></div>
  `;

  // Model selector
  let mhtml = '<table><thead><tr><th>ID</th><th>言語</th><th>サイズ</th><th>ステータス</th><th>リポジトリ</th><th></th></tr></thead><tbody>';
  Object.entries(moshi.models).forEach(([key, m]) => {
    const active = key === moshi.current;
    mhtml += `<tr style="${active ? 'background:rgba(168,85,247,.1)' : ''}">
      <td><b>${key}</b>${active ? ' <span style="color:#9333ea">◀現在</span>' : ''}</td>
      <td>${m.lang}</td>
      <td>${m.size}</td>
      <td>${m.status}</td>
      <td class="preview">${m.repo}</td>
      <td>${active ? '' : `<button data-model="${key}" onclick="switchModel('${key}')" style="padding:4px 12px;border:none;border-radius:8px;background:linear-gradient(135deg,#c084fc,#818cf8);color:#fff;cursor:pointer;font-size:.8rem">切替</button>`}</td>
    </tr>`;
  });
  mhtml += '</tbody></table>';
  document.getElementById('model-selector').innerHTML = mhtml;

  // TTS selector
  let thtml = '<table><thead><tr><th>ID</th><th>言語</th><th>サイズ</th><th>ステータス</th><th></th></tr></thead><tbody>';
  Object.entries(tts.models).forEach(([key, m]) => {
    const active = key === tts.current;
    thtml += `<tr style="${active ? 'background:rgba(168,85,247,.1)' : ''}">
      <td><b>${key}</b>${active ? ' <span style="color:#9333ea">◀現在</span>' : ''}</td>
      <td>${m.lang}</td>
      <td>${m.size}</td>
      <td>${m.status}</td>
      <td>${active ? '' : `<button onclick="switchTTS('${key}')" style="padding:4px 12px;border:none;border-radius:8px;background:linear-gradient(135deg,#c084fc,#818cf8);color:#fff;cursor:pointer;font-size:.8rem">切替</button>`}</td>
    </tr>`;
  });
  thtml += '</tbody></table>';
  document.getElementById('tts-selector').innerHTML = thtml;

  // User memory (RAG)
  const facts = mem.facts || [];
  document.getElementById('user-memory').innerHTML = facts.length === 0
    ? '<div class="empty">まだ記憶がありません。Pipeline モードで会話すると自動的に蓄積されます。</div>'
    : `<div style="margin-bottom:8px;font-size:.8rem;color:#9333ea">セッション数: ${mem.session_count} ／ 記憶: ${facts.length}件</div>`
      + facts.map(f => `<div class="kv"><span class="v">• ${f}</span></div>`).join('');

  // Variants
  let vhtml = '<table><thead><tr><th>ID</th><th>戦略</th><th>挨拶</th></tr></thead><tbody>';
  variants.variants.forEach(v=>{
    vhtml += `<tr><td><b>${v.id}</b></td><td>${v.strategy}</td><td class="preview">${v.greeting}</td></tr>`;
  });
  vhtml += '</tbody></table>';
  document.getElementById('variants').innerHTML = vhtml;
}
load();
</script></body></html>"""

# 挨拶バリアントの戦略ラベル（管理画面用）
GREETING_STRATEGIES = {
    "I1": "ネガティビティ（嫌いな食べ物）",
    "I2": "最近記憶（今日のイラッ）",
    "I3": "二択（朝型/夜型）",
    "I4": "パターン補完（もし休みなら）",
    "I5": "反論誘発（朝活いい説）",
    "I6": "具体エピソード（1週間の笑い）",
    "I7": "地味な自慢",
    "I8": "自己認識（めんどくさい性格）",
    "I9": "10点評価（コンディション）",
    "I10": "共感ムカつき",
    "I11": "あるある共感（日曜の憂鬱）",
    "I12": "最近記憶（買物）",
    "I13": "二択（旅行：計画/ノリ）",
    "I14": "連想（夏休みの匂い）",
    "I15": "愚痴誘発",
    "I16": "欲望直球（今夜何食べたい）",
    "I17": "達成感（最近やってよかったこと）",
    "I18": "タイムトラベル（10年前の自分が見たら）",
    "I19": "こだわり発掘（譲れないこと）",
    "I20": "ポジティブ記憶（ありがとうと言われた）",
    "I21": "天気メタファー（今日の気分）",
    "I22": "子ども時代の夢（なりたかった大人）",
    "I23": "最近の成功体験",
    "I24": "ちょっと刺さる質問（席を譲る派？）",
    "I25": "休日ルーティン",
    "I26": "最近ハマってること",
    "I27": "お仕事/学校の愚痴誘発",
    "I28": "自己認識（変わってる部分）",
    "I29": "人間観察（いいなと思う瞬間）",
    "I30": "最近泣いた？",
    "I31": "生まれ変わり妄想",
    "I32": "他者から見た自分（〜といえば）",
    "I33": "未達成リスト（今年やってないこと）",
    "I34": "秘密の食の好み",
    "I35": "内向/外向（充電できる時間）",
    "I36": "最後に本気で笑った瞬間",
    "I37": "人生の決断（あのとき決めてよかった）",
    "I38": "自己採点（何点？何引いた？）",
    "I39": "嫌いじゃないけど苦手",
    "I40": "親に怒られた記憶",
    "I41": "贅沢感（最近感じた贅沢）",
    "I42": "SNS羨ましさの正体",
    "I43": "誰にも勧めてないおすすめ",
    "I44": "急に休みになったら",
    "I45": "もし違う選択をしていたら",
    "I46": "認められていない努力",
    "I47": "久しぶりに連絡した人",
    "I48": "続けられてること",
    "I49": "最後の日に言いたいこと",
    "I50": "人間らしさを感じる瞬間",
}

async def admin_page(request):
    return web.Response(text=ADMIN_HTML, content_type="text/html",
                        headers={"Cache-Control": "no-store"})

async def variants_api(request):
    return web.json_response({
        "variants": [
            {"id": vid, "strategy": GREETING_STRATEGIES.get(vid, "-"), "greeting": g}
            for vid, g in GREETING_VARIANTS
        ]
    })

async def user_memory_api(request):
    if request.method == "DELETE":
        if USER_MEMORY_FILE.exists():
            USER_MEMORY_FILE.unlink()
        return web.json_response({"status": "cleared"})
    mem = {"facts": []}
    if USER_MEMORY_FILE.exists():
        with open(USER_MEMORY_FILE, encoding="utf-8") as f:
            mem = json.load(f)
    sessions = list(SESSIONS_DIR.glob("*.jsonl")) if SESSIONS_DIR.exists() else []
    return web.json_response({
        "facts": mem.get("facts", []),
        "session_count": len(sessions),
        "turn_counts": {s.stem: sum(1 for _ in open(s)) for s in sorted(sessions)[-10:]},
    })

async def export_training_api(request):
    """Phase3: セッションデータをmlx-lm LoRA形式にエクスポート"""
    if not SESSIONS_DIR.exists():
        return web.json_response({"error": "no sessions yet"})
    out_dir = Path("conversations/finetune")
    out_dir.mkdir(exist_ok=True)
    records = []
    for sf in sorted(SESSIONS_DIR.glob("*.jsonl")):
        turns = []
        with open(sf, encoding="utf-8") as f:
            for line in f:
                try: turns.append(json.loads(line))
                except: pass
        # 2turn以上のセッションのみ
        conv = [t for t in turns if t["role"] in ("user", "assistant")]
        if len(conv) < 4:
            continue
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for t in conv:
            messages.append({"role": t["role"], "content": t["text"]})
        records.append({"messages": messages})

    if not records:
        return web.json_response({"error": "no valid sessions"})

    import random
    random.shuffle(records)
    split = max(1, int(len(records) * 0.9))
    train, valid = records[:split], records[split:]

    for name, data in [("train.jsonl", train), ("valid.jsonl", valid)]:
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return web.json_response({
        "train": len(train),
        "valid": len(valid),
        "output_dir": str(out_dir),
        "next_step": f"mlx_lm.lora --model {MLX_LM_REPO} --data {out_dir} --train --iters 1000",
    })

# ── HTML ─────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>つながりAI</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=M+PLUS+Rounded+1c:wght@400;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{
  font-family:'M PLUS Rounded 1c','Hiragino Maru Gothic Pro',sans-serif;
  background:linear-gradient(135deg,#fce4f3 0%,#ede0ff 50%,#dbeafe 100%);
  display:flex;flex-direction:column;align-items:center;
}
h1{font-size:1.25rem;font-weight:700;color:#a855f7;letter-spacing:.05em;
   padding:14px 0 6px;text-shadow:0 2px 8px rgba(168,85,247,.2);flex-shrink:0}
/* orb */
.orb-wrap{position:relative;width:110px;height:110px;margin-bottom:8px;flex-shrink:0}
.orb-ring{position:absolute;border-radius:50%;border:2px solid rgba(168,85,247,.25);animation:pulse 2.4s ease-in-out infinite}
.orb-ring:nth-child(1){inset:0;animation-delay:0s}
.orb-ring:nth-child(2){inset:8px;animation-delay:.4s}
.orb-ring:nth-child(3){inset:16px;animation-delay:.8s}
.orb-core{position:absolute;inset:26px;border-radius:50%;
  background:linear-gradient(135deg,#f472b6,#a78bfa);
  box-shadow:0 0 28px rgba(167,139,250,.6),0 0 56px rgba(244,114,182,.3);
  display:flex;align-items:center;justify-content:center;font-size:1.8rem}
.orb-core.speaking{animation:speakPulse .5s ease-in-out infinite alternate}
@keyframes pulse{0%,100%{transform:scale(1);opacity:.4}50%{transform:scale(1.08);opacity:.9}}
@keyframes speakPulse{from{transform:scale(1)}to{transform:scale(1.13);box-shadow:0 0 56px rgba(244,114,182,.9)}}
/* mode tabs */
.tabs{display:flex;gap:6px;margin-bottom:8px;flex-shrink:0}
.tab{padding:5px 16px;border-radius:20px;border:none;cursor:pointer;font-size:.8rem;
     font-family:inherit;background:rgba(255,255,255,.5);color:#7c3aed;transition:.2s}
.tab.active{background:linear-gradient(135deg,#c084fc,#818cf8);color:#fff;font-weight:700}
/* status */
.status{font-size:.78rem;color:#9333ea;min-height:1.1em;margin-bottom:6px;flex-shrink:0}
/* chat */
.chat{width:100%;max-width:520px;flex:1;overflow-y:auto;
      display:flex;flex-direction:column;align-items:center;gap:10px;padding:0 16px 88px}
.bubble{width:100%;max-width:420px;padding:12px 18px;border-radius:18px;
        font-size:.92rem;line-height:1.6;animation:fadeUp .3s ease;
        word-break:break-word;white-space:pre-wrap}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.bubble.ai{background:rgba(255,255,255,.88);backdrop-filter:blur(8px);
            border:1px solid rgba(168,85,247,.2);color:#4b2069}
.bubble.user{background:linear-gradient(135deg,#c084fc,#818cf8);color:#fff}
/* buttons */
.mic-btn{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);
  width:62px;height:62px;border-radius:50%;border:none;cursor:pointer;
  background:linear-gradient(135deg,#f472b6,#a78bfa);
  box-shadow:0 4px 24px rgba(167,139,250,.5);font-size:1.5rem;
  display:flex;align-items:center;justify-content:center;transition:.15s;z-index:10}
.mic-btn:hover{transform:translateX(-50%) scale(1.08)}
.mic-btn.active{background:linear-gradient(135deg,#fb7185,#f472b6)}
.settings-btn{position:fixed;top:14px;right:14px;width:38px;height:38px;
  border-radius:50%;border:none;cursor:pointer;z-index:200;
  background:rgba(255,255,255,.85);backdrop-filter:blur(8px);
  box-shadow:0 2px 12px rgba(168,85,247,.25);font-size:1.1rem;
  display:flex;align-items:center;justify-content:center;transition:.2s}
.settings-btn:hover{transform:rotate(60deg)}
/* modal */
.overlay{display:none;position:fixed;inset:0;background:rgba(80,30,120,.2);
  backdrop-filter:blur(4px);z-index:300;align-items:center;justify-content:center}
.overlay.open{display:flex}
.modal{background:rgba(255,255,255,.96);border-radius:24px;padding:26px 26px 20px;
  max-width:360px;width:90%;box-shadow:0 8px 40px rgba(168,85,247,.3);animation:fadeUp .25s ease}
.modal h2{font-size:1.05rem;color:#7c3aed;margin-bottom:16px}
.srow{display:flex;flex-direction:column;gap:3px;margin-bottom:12px}
.srow label{font-size:.7rem;color:#9333ea;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
.srow span{font-size:.83rem;color:#3b0764;background:#f5f0ff;border-radius:8px;padding:5px 10px;word-break:break-all}
.modal-close{display:block;width:100%;margin-top:6px;padding:10px;border:none;border-radius:12px;
  background:linear-gradient(135deg,#c084fc,#818cf8);color:#fff;font-size:.9rem;
  cursor:pointer;font-family:inherit}
/* start overlay */
#startOverlay{position:fixed;inset:0;z-index:400;
  background:linear-gradient(135deg,#fce4f3 0%,#ede0ff 50%,#dbeafe 100%);
  display:flex;align-items:center;justify-content:center}
#startCard{text-align:center;padding:40px 36px;
  background:rgba(255,255,255,.85);backdrop-filter:blur(12px);
  border-radius:32px;box-shadow:0 12px 60px rgba(168,85,247,.25);
  max-width:340px;width:90%;animation:fadeUp .6s ease}
.start-orb{font-size:3.5rem;margin-bottom:16px;display:inline-block;
  animation:pulse 2.2s ease-in-out infinite}
#startCard h2{font-size:1.5rem;color:#7c3aed;margin-bottom:8px;font-weight:700}
#startCard p{font-size:.88rem;color:#6d28d9;margin-bottom:28px;line-height:1.6}
#startBtn{padding:14px 40px;border:none;border-radius:999px;cursor:pointer;
  background:linear-gradient(135deg,#f472b6,#a78bfa);color:#fff;
  font-size:1.05rem;font-weight:700;font-family:inherit;
  box-shadow:0 4px 24px rgba(167,139,250,.5);transition:.2s}
#startBtn:hover{transform:scale(1.06);box-shadow:0 6px 32px rgba(244,114,182,.6)}
.cursor{display:inline-block;width:2px;background:#a855f7;
  margin-left:1px;animation:blink .7s step-end infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
</style>
</head>
<body>

<div id="startOverlay">
  <div id="startCard">
    <div class="start-orb">💜</div>
    <h2>つながりAI</h2>
    <p>36の質問で、あなたと深くつながろう<br>科学的に親密さを育む会話体験</p>
    <button id="startBtn">はじめる ✨</button>
  </div>
</div>

<button class="settings-btn" id="settingsBtn">⚙️</button>

<div class="overlay" id="overlay">
  <div class="modal">
    <h2>⚙️ 現在の設定</h2>
    <div class="srow"><label>モード</label><span id="s-mode">-</span></div>
    <div class="srow"><label>S2Sモデル</label><span id="s-moshi">-</span></div>
    <div class="srow"><label>STT（音声認識）</label><span id="s-stt">-</span></div>
    <div class="srow"><label>LLM（言語モデル）</label><span id="s-llm">-</span></div>
    <div class="srow"><label>TTS（音声合成）</label><span id="s-tts">-</span></div>
    <div class="srow"><label>TTSモデル</label><span id="s-tts-model">-</span></div>
    <div class="srow"><label>VAD 無音判定</label><span id="s-vad">-</span></div>
    <div class="srow"><label>サンプルレート</label><span id="s-rate">-</span></div>
    <a href="/admin" target="_blank" style="display:block;margin-top:6px;padding:10px;border-radius:12px;background:rgba(168,85,247,.12);color:#7c3aed;font-size:.88rem;text-align:center;text-decoration:none;font-weight:700">📊 管理画面を開く →</a>
    <button class="modal-close" id="modalClose">閉じる</button>
  </div>
</div>

<h1>💜 つながりAI</h1>
<div class="orb-wrap">
  <div class="orb-ring"></div><div class="orb-ring"></div><div class="orb-ring"></div>
  <div class="orb-core" id="orb">✨</div>
</div>
<div class="tabs" style="display:none">
  <button class="tab" id="tabPipeline" onclick="setMode('pipeline')">🧠 Pipeline</button>
  <button class="tab active" id="tabMoshi" onclick="setMode('moshi')">🎙️ Moshi（日本語S2S）</button>
</div>
<div class="status" id="status">接続中...</div>
<div class="chat" id="chat"></div>
<button class="mic-btn active" id="micBtn">🎙️</button>

<script>
const chat = document.getElementById('chat');
const status = document.getElementById('status');
const orb = document.getElementById('orb');
const micBtn = document.getElementById('micBtn');

let ws, audioCtx, processor, source, stream;
let micActive = false;
let currentMode = 'pipeline';
// Moshi playback
let moshiCtx = null, moshiPlayTime = 0;

function typewriter(el, text, speed) {
  speed = speed || 28;
  return new Promise(function(resolve) {
    var i = 0;
    var cursor = document.createElement('span');
    cursor.className = 'cursor';
    el.appendChild(cursor);
    var iv = setInterval(function() {
      cursor.before(document.createTextNode(text[i++]));
      chat.scrollTop = chat.scrollHeight;
      if (i >= text.length) { clearInterval(iv); cursor.remove(); resolve(); }
    }, speed);
  });
}

function addBubble(role, text) {
  const d = document.createElement('div');
  d.className = 'bubble ' + role;
  d.textContent = '';
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  if (role === 'ai') {
    typewriter(d, text);
  } else {
    d.textContent = text;
    chat.scrollTop = chat.scrollHeight;
  }
}

function playMoshiPCM(buf) {
  if (!moshiCtx) { moshiCtx = new AudioContext({sampleRate: 24000}); moshiPlayTime = 0; }
  const pcm = new Float32Array(buf);
  const ab = moshiCtx.createBuffer(1, pcm.length, 24000);
  ab.copyToChannel(pcm, 0);
  const src = moshiCtx.createBufferSource();
  src.buffer = ab; src.connect(moshiCtx.destination);
  const now = moshiCtx.currentTime;
  moshiPlayTime = Math.max(now, moshiPlayTime);
  src.start(moshiPlayTime);
  moshiPlayTime += ab.duration;
}

function connect() {
  ws = new WebSocket('ws://' + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { status.textContent = '聞いています...'; startMic(); };
  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) { playMoshiPCM(e.data); return; }
    const msg = JSON.parse(e.data);
    if (msg.type === 'ai')      addBubble('ai', msg.text);
    if (msg.type === 'user')    addBubble('user', msg.text);
    if (msg.type === 'status')  status.textContent = msg.text;
    if (msg.type === 'speaking') {
      orb.classList.toggle('speaking', msg.on);
      orb.textContent = msg.on ? '🗣️' : '✨';
      if (msg.on && moshiCtx) moshiPlayTime = moshiCtx.currentTime; // Moshiキューをフラッシュ
    }
  };
  ws.onclose = () => { status.textContent = '再接続中...'; setTimeout(connect, 2000); };
}

async function startMic() {
  if (micActive) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: {channelCount:1} });
    audioCtx = new AudioContext({sampleRate: 16000});
    source = audioCtx.createMediaStreamSource(stream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== 1) return;
      ws.send(new Float32Array(e.inputBuffer.getChannelData(0)).buffer);
    };
    source.connect(processor);
    processor.connect(audioCtx.destination);
    micActive = true;
    micBtn.classList.add('active');
  } catch(err) {
    status.textContent = 'マイクを許可してください';
  }
}

micBtn.onclick = () => {
  if (!micActive) { startMic(); }
  else {
    processor?.disconnect(); source?.disconnect();
    stream?.getTracks().forEach(t => t.stop());
    audioCtx?.close(); audioCtx = null;
    micActive = false; micBtn.classList.remove('active');
    status.textContent = 'マイク停止中';
  }
};

async function setMode(mode) {
  currentMode = mode;
  document.getElementById('tabPipeline').classList.toggle('active', mode === 'pipeline');
  document.getElementById('tabMoshi').classList.toggle('active', mode === 'moshi');
  await fetch('/api/mode', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({mode})});
  ws?.close();
}

// Settings
const overlay = document.getElementById('overlay');
document.getElementById('settingsBtn').addEventListener('click', async () => {
  const s = await fetch('/api/settings?t=' + Date.now()).then(r => r.json());
  document.getElementById('s-mode').textContent = s.mode;
  document.getElementById('s-moshi').textContent = s.moshi_model + ' (' + s.moshi_repo + ')';
  document.getElementById('s-stt').textContent = s.stt_model;
  document.getElementById('s-llm').textContent = s.llm_model;
  document.getElementById('s-tts').textContent = s.tts_voice;
  document.getElementById('s-tts-model').textContent = s.tts_model + ' (' + s.tts_mode + ')';
  document.getElementById('s-vad').textContent = s.vad_silence_sec + '秒 / 閾値 ' + s.vad_threshold;
  document.getElementById('s-rate').textContent = s.sample_rate + ' Hz';
  overlay.classList.add('open');
});
document.getElementById('modalClose').addEventListener('click', () => overlay.classList.remove('open'));
overlay.addEventListener('click', e => { if (e.target === overlay) overlay.classList.remove('open'); });

// Start button
document.getElementById('startBtn').addEventListener('click', function() {
  document.getElementById('startOverlay').style.display = 'none';
  connect();
});
</script>
</body>
</html>"""

async def index(request):
    return web.Response(text=HTML, content_type="text/html",
                        headers={"Cache-Control": "no-store"})

_model_ready = False

async def main():
    global loop, _model_ready
    loop = asyncio.get_event_loop()

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/settings", settings_api)
    app.router.add_post("/api/mode", mode_api)
    app.router.add_get("/api/moshi_model", moshi_model_api)
    app.router.add_post("/api/moshi_model", moshi_model_api)
    app.router.add_get("/api/tts_model", tts_model_api)
    app.router.add_post("/api/tts_model", tts_model_api)
    app.router.add_get("/api/ab_results", ab_results_api)
    app.router.add_get("/api/variants", variants_api)
    app.router.add_get("/api/user_memory", user_memory_api)
    app.router.add_delete("/api/user_memory", user_memory_api)
    app.router.add_get("/api/export_training", export_training_api)
    app.router.add_get("/admin", admin_page)

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", PORT).start()

    print(f"\n✓ サーバー起動 → http://localhost:{PORT}")
    print(f"  ブラウザで開いてください: http://localhost:{PORT}\n")

    # モデルはバックグラウンドでロード
    def _load():
        global _model_ready
        load_pipeline_models()
        _model_ready = True
    threading.Thread(target=_load, daemon=True).start()

    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n終了しました。")
