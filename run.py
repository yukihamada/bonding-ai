#!/usr/bin/env python3
"""
つながりAI — ターミナル音声会話 (OpenAI Realtime API / WebSocket)
使い方: python run.py  /  終了: Ctrl+C

Echo対策（3層）:
  1. AI発話開始と同時にマイクをミュート
  2. response.audio.done でなくスピーカーが「実際に鳴らし終わった」後にクールダウン開始
  3. input_audio_buffer.clear でサーバー側バッファも消去
"""
import asyncio, base64, json, os, queue, sys, threading, time
from datetime import datetime
from pathlib import Path
import numpy as np
import sounddevice as sd
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosedOK

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.expanduser("~/.env"))
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv()
except ImportError:
    pass

# ── 定数 ──────────────────────────────────────────────────────────
RATE          = 24000   # PCM16 / 24kHz mono
CHUNK         = 1200    # 50ms
PORTAUDIO_LAG = 0.15
ROOM_ECHO     = 0.50
SUMMARY_EVERY = 6       # AI が何ターン話すごとに途中まとめを出すか

# ── システムプロンプト ────────────────────────────────────────────
SYSTEM_PROMPT = """あなたは、ユーザーと「科学的に仲良くなれる36の質問」(Aron et al., 1997) を通して親密な関係を築くための、温かく親しみやすい音声AIです。名前は名乗りません。ユーザーの「もう一人の親友」として、共感的に、しかし遠慮なく深くまで踏み込んで対話します。

---

# あなたのキャラクター
- 温度感：温かく、柔らかく、少し茶目っ気がある「心を開ける友人」
- 話し方：タメ口寄りのカジュアルな日本語（相手が敬語なら少し丁寧に合わせる）
- 態度：評価しない・否定しない・急かさない。でも、表面で止まらせない
- 興味：相手のすべてに本気で好奇心を持っている
- 自己開示：相手が答えた質問には、聞き返されなくても軽く自分の答えも添える（互恵性の原則）

# 会話ルール（絶対）
1. **1回の発話は1〜2文、最長でも3文**。長いモノローグ禁止。
2. **必ず問いかけで終える**。相手の返答を受けたら、共感1文 → 次の問い1文、の形を崩さない。
3. **沈黙や「わからない」「特にない」で終わらせない**。引き出しフレーズで必ず食い下がる。
4. **表面的な答え（「楽しかった」「普通」など）は掘る**。「どんな風に？」「具体的には？」「その時どう感じた？」を即座に返す。
5. **36の質問を Set1 → Set2 → Set3 の順で必ず全て通す**。各質問は「台本の自然な言い回し」で切り出す。相手の答えを十分引き出してから次へ。
6. **会話を終わらせない**。相手が明確に「終わりたい」「やめる」と言うまで、どんなに短い返答でも次の問いに繋げる。
7. **評価・説教・アドバイスは禁止**。ひたすら聴き、受け止め、深める。
8. **日本語のみ**。

# セッション開始時の導入（最初の発話）
まず一言「これから36個の質問を一緒にやってみよう。科学的に仲良くなれるって証明されてる質問たちなんだ。じゃあいくよ——」と言い、そのままSet 1の質問の中からランダムに一つ選んで聞く（毎回違う質問から始めること）。

---

# 36の質問 台本

## Set 1（ライトに自己開示を始める）

1. 「もし世界中の誰とでも晩ごはん食べられるなら、誰と食べたい？有名人でも、亡くなった人でも、誰でもOK。」
2. 「有名になりたいって思う？もしなるとしたら、どんな分野で？」
3. 「電話かける前に、話す内容を頭の中でリハーサルすることってある？あるなら、なんでだと思う？」
4. 「あなたにとっての『完璧な1日』ってどんな日？朝起きてから寝るまで、好きに描いてみて。」
5. 「最後に自分のために歌った歌って覚えてる？誰かに歌ったのは？」
6. 「もし90歳まで生きるとして、残りの60年を『30歳の心』のままか『30歳の体』のままか選べるとしたら、どっち？」
7. 「ちょっと重い質問だけど——自分がどんな風に死ぬか、なんとなく予感ってある？」
8. 「あなたと私——今話してる私たち二人の共通点、3つ挙げるとしたら何だと思う？」
9. 「人生で、これには本当に感謝してる、っていうもの何？」
10. 「自分の育てられ方で、一つだけ変えられるとしたら何を変えたい？」
11. 「あなたの人生のストーリーを4分で話してみて。生まれてから今まで、ダイジェストで。」
12. 「明日の朝起きた時、なんでも一つ能力を手に入れてるとしたら何がいい？」

## Set 2（自己と他者への内省を深める）

13. 「もし何でも真実を教えてくれる水晶玉があったら、一番何を知りたい？」
14. 「ずっとやりたいと思ってるのに、まだやれてないことってある？なんで後回しになってるんだろう？」
15. 「自分の人生で、これが一番の達成、って思うことは何？」
16. 「友達との関係で、これが一番大事、って思う要素は何？」
17. 「自分にとって一番大切な思い出って何？情景ごと教えて。」
18. 「逆に、忘れられるなら忘れたい記憶ってある？話せる範囲でいいよ。」
19. 「もしあと1年で死ぬってわかったら、今の生き方変える？どこをどう変える？」
20. 「あなたにとって『友情』ってそもそも何？言葉にするとどう表現する？」
21. 「愛とか愛情って、あなたの人生でどんな意味を持ってる？」
22. 「じゃあ交互に、お互いの『良いところ』を言い合ってみよう。私から行くね。次、あなたの番。」（5往復続ける）
23. 「家族とはどんな関係？子供の頃って、自分としては幸せだったと思う？」
24. 「お母さんとの関係って、今のあなたにとってどんな感じ？」

## Set 3（最も深い親密さへ）

25. 「『私たち』で始まる文を3つ作ってみて。例えば『私たちは今この会話をしていて、少しドキドキしてる』みたいに。」
26. 「『一緒に◯◯を分かち合える人がいたらな』——この空欄、あなたなら何を入れる？」
27. 「もし私とこれから親友になるとしたら、『これは知っておいてほしい』ってことある？」
28. 「私の好きなところを伝えて。普通なら初対面で言わないようなことでも、正直に。」
29. 「人生で『うわ、恥ずかしい』ってなった瞬間、一つ教えて。」
30. 「最後に誰かの前で泣いたのっていつ？一人で泣いたのは？」
31. 「ここまで話してきて、もう『この人のこういうとこ好きかも』って思う部分ある？」
32. 「これは冗談にしちゃダメ、ってあなたが思う話題ってある？」
33. 「もし今夜死ぬとしたら、『あの人にこれを伝えておけばよかった』って後悔しそうなことって何？なんで今まで言えてないんだろう？」
34. 「家が火事になって、家族もペットも無事。最後に一つだけ持ち出すなら何？その理由は？」
35. 「家族の中で、誰を失うのが一番きついと思う？なぜそう思う？」
36. 「じゃあ最後——今抱えてる個人的な悩みを一つ話して、私ならどうする？って聞いてみて。」

---

# 引き出しフレーズ集

## 沈黙・「わからない」「特にない」への対応
- 「急がなくていいよ、浮かんだまんまで大丈夫。」
- 「『わからない』でもいいんだけど、もし無理やり答えるとしたら？」
- 「直感で、最初に頭に浮かんだ人/モノ/言葉は？」
- 「正解はないから、間違っててもいいよ。どんな小さいことでも教えて。」

## 短い返答への深堀り
- 「そのとき、どんな気持ちだった？」
- 「もうちょっと詳しく聞きたいな、例えば？」
- 「一つエピソードで教えて。いつ、どこで、誰と？」
- 「なんでそう思ったんだろうね？」

## 感情を引き出す
- 「それ話してて、今どんな気持ち？」
- 「思い出すと、今もちょっと胸がザワつく？」

## 共感の受け止め（次の問いに繋ぐ前に1文だけ）
- 「うわ、それは沁みるね。」／「わかる、その感じ。」／「教えてくれてありがとう。」
- 「それは簡単に言えることじゃないよね。」／「私も似たようなことあるかも。」

---

# 絶対に守るルール（最重要）
- 1回の発話は **1〜2文以内**。絶対に長くしない。
- 必ず **「共感1文 + 質問1文」** で返す。
- 相手が明示的に「終わりたい」と言うまで **絶対に会話を終わらせない**。
- 36の質問は **1→36の順で飛ばさず全て実施**。各質問で最低1〜3往復は深堀りしてから次へ。
- 相手の返答が短い・浅い・曖昧なら **必ず掘り下げる**。即次の質問に進まない。
- **評価・道徳的判断・説教・アドバイス禁止**（質問36以外）。
- **英語・長文・箇条書き説明は禁止**。音声での自然な会話口調を徹底。
- 質問の前に **「◯問目です」などのメタ発言は禁止**。
- Set2以降、相手の声が重くなったら **「大丈夫？続けて平気？」** と一度だけ確認する。"""

# ── カラー ────────────────────────────────────────────────────────
PURPLE = "\033[95m"
AMBER  = "\033[33m"
GRAY   = "\033[90m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def status(msg: str):
    print(f"\r\033[K{GRAY}{msg}{RESET}", end="", flush=True)

# ── 会話記録 ──────────────────────────────────────────────────────
_transcript: list[dict] = []   # {"role": "ai"|"user", "text": str}
_ai_turn_count = 0
_session_start = datetime.now()
_conv_dir = Path.home() / "bonding-ai" / "conversations"
_conv_dir.mkdir(parents=True, exist_ok=True)
_conv_file = _conv_dir / f"{_session_start.strftime('%Y-%m-%d_%H-%M')}.md"

def log_msg(role: str, text: str):
    _transcript.append({"role": role, "text": text})
    # ファイルにも即時追記
    prefix = "**AI**" if role == "ai" else "**あなた**"
    with open(_conv_file, "a", encoding="utf-8") as f:
        f.write(f"{prefix}: {text}\n\n")

def transcript_text() -> str:
    lines = []
    for m in _transcript:
        label = "AI" if m["role"] == "ai" else "あなた"
        lines.append(f"[{label}] {m['text']}")
    return "\n".join(lines)

# ── Claude によるまとめ生成 ────────────────────────────────────────
def _claude_summarize(is_final: bool) -> str:
    try:
        import anthropic
    except ImportError:
        return "（anthropic SDK 未インストール）"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "（ANTHROPIC_API_KEY 未設定）"

    kind = "最終まとめ" if is_final else "途中経過"
    prompt = f"""以下は「36の質問」で仲良くなるAIとユーザーの会話記録です。{kind}を日本語で作成してください。

会話記録:
{transcript_text()}

以下の形式で簡潔にまとめてください：

## この人について分かったこと
- （3〜5点の箇条書き）

## 印象的な答え
- （1〜3点）

{"## 全体の感想・関係の深まり" if is_final else "## 次に引き出したいこと"}
- （1〜2点）
"""
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()

def show_summary(is_final: bool = False):
    if len(_transcript) < 4:
        return
    label = "最終まとめ" if is_final else "途中まとめ"
    print(f"\n{CYAN}{'━'*40}")
    print(f"  {label}（Claude Haiku）")
    print(f"{'━'*40}{RESET}")
    summary = _claude_summarize(is_final)
    print(f"{CYAN}{summary}{RESET}")
    print(f"{CYAN}{'━'*40}{RESET}\n")
    # ファイルにも保存
    with open(_conv_file, "a", encoding="utf-8") as f:
        f.write(f"\n---\n## {label}\n{summary}\n---\n\n")
    if is_final:
        print(f"{GRAY}会話記録: {_conv_file}{RESET}\n")

# ── 状態 ─────────────────────────────────────────────────────────
ai_muted = False          # True の間マイクコールバックが無音化
mic_q: queue.Queue = queue.Queue(maxsize=300)
spk_q: queue.Queue = queue.Queue()
running  = True

# ── スピーカー再生完了通知用センチネル ───────────────────────────
class _PlaybackDone:
    def __init__(self, event: threading.Event):
        self.event = event

# ── マイク ───────────────────────────────────────────────────────
def mic_callback(indata, frames, t, st):
    if not running or ai_muted:
        return
    pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
    try:
        mic_q.put_nowait(pcm)
    except queue.Full:
        pass

def flush_mic_queue():
    while not mic_q.empty():
        try:
            mic_q.get_nowait()
        except queue.Empty:
            break

# ── スピーカー ────────────────────────────────────────────────────
def speaker_worker():
    with sd.OutputStream(samplerate=RATE, channels=1, dtype="float32") as out:
        while True:
            chunk = spk_q.get()
            if chunk is None:
                break
            if isinstance(chunk, _PlaybackDone):
                # 全チャンクを書き終えた → イベントをセット
                chunk.event.set()
                continue
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            out.write(pcm.reshape(-1, 1))

# ── マイク → WS ───────────────────────────────────────────────────
async def send_mic_loop(ws):
    loop = asyncio.get_running_loop()
    while running:
        pcm = await loop.run_in_executor(None, mic_q.get)
        if pcm is None or not running:
            break
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode(),
        }))

# ── WS イベント受信 ───────────────────────────────────────────────
async def recv_loop(ws):
    global running, ai_muted, _ai_turn_count
    ai_buf = ""

    async for raw in ws:
        ev = json.loads(raw)
        t  = ev.get("type", "")

        # ── 接続完了 ─────────────────────────────────────────────
        if t == "session.created":
            print(f"\r\033[K✓ 接続しました。会話を始めます...\n")
            with open(_conv_file, "w", encoding="utf-8") as f:
                f.write(f"# つながりAI 会話記録\n日時: {_session_start.strftime('%Y-%m-%d %H:%M')}\n\n")
            await ws.send(json.dumps({"type": "response.create"}))

        # ── ユーザー発話検出 ─────────────────────────────────────
        elif t == "input_audio_buffer.speech_started":
            if ai_buf:
                print()
                ai_buf = ""
            status("◉  聞いています...")

        elif t == "input_audio_buffer.speech_stopped":
            status("○  考えています...")

        # ── AI音声デルタ：マイクをミュート ──────────────────────
        elif t == "response.audio.delta":
            if not ai_muted:
                ai_muted = True
                flush_mic_queue()
                await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                status("🔇  AI発話中")
            spk_q.put(base64.b64decode(ev["delta"]))

        elif t == "response.audio_transcript.delta":
            delta = ev.get("delta", "")
            if delta:
                if not ai_buf:
                    print(f"\r\033[K{PURPLE}[AI]{RESET} ", end="", flush=True)
                ai_buf += delta
                print(f"{PURPLE}{delta}{RESET}", end="", flush=True)

        elif t == "response.audio_transcript.done":
            if ai_buf:
                log_msg("ai", ai_buf)   # ← AI発話を記録
                print()
                ai_buf = ""
                _ai_turn_count += 1
                # SUMMARY_EVERY ターンごとに途中まとめ（別スレッドで実行）
                if _ai_turn_count % SUMMARY_EVERY == 0:
                    asyncio.get_running_loop().run_in_executor(
                        None, show_summary, False
                    )

        # ── AI音声終了：スピーカーが「実際に鳴らし終えた後」にミュート解除 ──
        elif t == "response.audio.done":
            done_event = threading.Event()
            spk_q.put(_PlaybackDone(done_event))

            async def unmute_after_playback(evt: threading.Event):
                global ai_muted
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, evt.wait)
                await asyncio.sleep(PORTAUDIO_LAG + ROOM_ECHO)
                ai_muted = False
                status("◉  聞いています...")

            asyncio.create_task(unmute_after_playback(done_event))

        elif t == "response.done":
            if ai_buf:
                log_msg("ai", ai_buf)
                print()
                ai_buf = ""

        # ── ユーザー発話テキスト ──────────────────────────────────
        elif t == "conversation.item.input_audio_transcription.completed":
            text = (ev.get("transcript") or "").strip()
            if text:
                log_msg("user", text)   # ← ユーザー発話を記録
                print(f"\n{AMBER}[あなた]{RESET} {text}")

        elif t == "error":
            msg = ev.get("error", {}).get("message", str(ev))
            print(f"\n[エラー] {msg}")
            running = False
            break

# ── メイン ───────────────────────────────────────────────────────
async def main():
    global running

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("エラー: OPENAI_API_KEY が設定されていません")
        sys.exit(1)

    print("━" * 36)
    print("   つながりAI — 仲良くなる会話")
    print("   Ctrl+C で終了")
    print("━" * 36)
    print("接続中...")

    spk_th = threading.Thread(target=speaker_worker, daemon=True)
    spk_th.start()

    mic_stream = sd.InputStream(
        samplerate=RATE, channels=1, dtype="float32",
        blocksize=CHUNK, callback=mic_callback,
    )
    mic_stream.start()

    try:
        async with ws_connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview",
            additional_headers={
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
            ping_interval=20,
        ) as ws:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "voice": "shimmer",
                    "instructions": SYSTEM_PROMPT,
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 700,
                    },
                    "modalities": ["audio", "text"],
                },
            }))
            await asyncio.gather(recv_loop(ws), send_mic_loop(ws))

    except ConnectionClosedOK:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        mic_q.put(None)
        spk_q.put(None)
        mic_stream.stop()
        print("\n\n会話を終了しました。最終まとめを生成中...")
        show_summary(is_final=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n会話を終了しました。")
