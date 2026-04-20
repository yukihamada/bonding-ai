#!/usr/bin/env python3
"""
パイプライン自動テスト v2
edge-tts → soundfile → mlx-whisper → Qwen3.5 → 品質チェック（ユーモア・感情・精度）
"""
import asyncio, os, re, sys, tempfile
import numpy as np
import soundfile as sf
import edge_tts, mlx_whisper

sys.path.insert(0, os.path.dirname(__file__))

WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
MLX_LM_REPO  = "mlx-community/Qwen3.5-122B-A10B-4bit"
TTS_VOICE    = "ja-JP-NanamiNeural"

# ── テスト発話セット ──────────────────────────────────────────────
# (text, category)
TEST_UTTERANCES = [
    # 基本
    ("えーと、好きな食べ物はラーメンかな。",           "basic"),
    ("うーん、特にないかも。",                        "vague"),
    ("子供の頃に転校が多くて、友達作るのが苦手だったんだよね。", "emotional"),
    ("一番大切にしてることは、正直でいること、かな。",   "basic"),
    # 短答・困った系
    ("わからない。",                                 "short"),
    ("うん。",                                      "short"),
    ("まあ、そうかな。",                             "short"),
    # ユーモア系
    ("えっと、誰だろう。スティーブ・ジョブズとかかな？面白そうじゃん。", "funny"),
    ("完璧な一日ってさ、朝ゆっくり起きて、友達とごはん食べて、夜に映画見る感じかな。", "basic"),
    ("もし明日死ぬなら、ピザ食べて寝る笑。",           "funny"),
    # 感情系
    ("おじいちゃんかな。もう亡くなっちゃったんだけど。", "emotional_heavy"),
    ("一番の思い出は、父親と海に行った夏かな。",        "emotional"),
    # 深い系
    ("愛って、結局は選び続けることだと思う。",          "deep"),
    ("そうだなあ、やっぱり家族との思い出が一番かも。",   "basic"),
]

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"

async def tts_to_wav(text: str, voice: str = TTS_VOICE) -> np.ndarray:
    c = edge_tts.Communicate(text, voice=voice, rate="+5%")
    mp3_data = b""
    async for chunk in c.stream():
        if chunk["type"] == "audio":
            mp3_data += chunk["data"]
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_data); fname = f.name
    audio, sr = sf.read(fname)
    os.unlink(fname)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = np.gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g)
    return audio.astype(np.float32)

def whisper_transcribe(audio: np.ndarray) -> str:
    result = mlx_whisper.transcribe(
        audio, path_or_hf_repo=WHISPER_REPO, language="ja",
        initial_prompt="日本語の会話です。句読点を正確に付けてください。",
        word_timestamps=False,
    )
    return result.get("text", "").strip()

def cer(ref: str, hyp: str) -> float:
    ref = re.sub(r'\s+', '', ref)
    hyp = re.sub(r'\s+', '', hyp)
    if not ref:
        return 0.0
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if ref[i-1] == hyp[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n] / len(ref)

def llm_respond(text: str, model, tokenizer, history=None) -> str:
    from mlx_lm import generate as mlx_generate
    from app import SYSTEM_PROMPT
    if history is None:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text}]
    else:
        msgs = history + [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    reply = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=120, verbose=False)
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
    return reply

def check_quality(reply: str, category: str) -> tuple[bool, list[str]]:
    issues = []
    # 共通チェック
    ends_with_q  = reply.endswith("？") or reply.endswith("?") or "？" in reply[-8:]
    too_long     = len(reply) > 130
    is_japanese  = bool(re.search(r'[ぁ-んァ-ン]', reply))
    too_formal   = bool(re.search(r'素晴らしいですね|感動的です|ございます|でしょうか。$', reply))
    # カテゴリ別チェック
    has_casual   = bool(re.search(r'じゃん|だよね|かな|なの？|笑|わかる|だわ|いいね', reply))
    has_empathy  = bool(re.search(r'ありがとう|大変|つらい|すごい|いいな|だね', reply))

    if not ends_with_q:  issues.append("問いかけで終わっていない")
    if too_long:         issues.append(f"長すぎ({len(reply)}文字)")
    if not is_japanese:  issues.append("日本語でない")
    if too_formal:       issues.append("過剰に丁寧すぎる")

    # カテゴリ別加点/減点
    bonus = ""
    if category == "funny" and has_casual:   bonus = " [ユーモア対応✓]"
    if category in ("emotional", "emotional_heavy") and has_empathy: bonus = " [共感対応✓]"
    if category == "short" and ends_with_q:  bonus = " [短答引き出し✓]"

    ok = ends_with_q and not too_long and is_japanese and not too_formal
    return ok, issues, bonus

async def main():
    print(f"\n{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  つながりAI パイプラインテスト v2{RESET}")
    print(f"{BOLD}{'='*58}{RESET}\n")

    # ── Phase 1: TTS → Whisper ────────────────────────────────────
    print(f"{CYAN}{BOLD}Phase 1: TTS → Whisper 精度テスト（{len(TEST_UTTERANCES)}発話）{RESET}\n")
    stt_results = []
    for utt, cat in TEST_UTTERANCES:
        print(f"  [{cat}] 入力: 「{utt}」")
        audio = await tts_to_wav(utt)
        recognized = whisper_transcribe(audio)
        score = cer(utt, recognized)
        ok = score < 0.15
        mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"         認識: 「{recognized}」")
        print(f"         CER : {score:.1%}  {mark}\n")
        stt_results.append((utt, recognized, score, ok, cat))

    avg_cer   = np.mean([r[2] for r in stt_results])
    pass_rate = sum(r[3] for r in stt_results) / len(stt_results)
    print(f"{BOLD}Phase 1: 平均CER {avg_cer:.1%} / 合格率 {pass_rate:.0%}{RESET}\n")

    # カテゴリ別集計
    cats = {}
    for _, _, score, ok, cat in stt_results:
        cats.setdefault(cat, []).append(score)
    for cat, scores in sorted(cats.items()):
        print(f"  {cat:20s}: avg CER {np.mean(scores):.1%}")
    print()

    # ── Phase 2: LLM 応答品質テスト ──────────────────────────────
    print(f"{CYAN}{BOLD}Phase 2: LLM 応答品質テスト（ユーモア・共感・短答）{RESET}\n")
    print("  Qwen3.5 ロード中...")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(MLX_LM_REPO)
    print("  ✓ ロード完了\n")

    llm_results = []
    for utt, recognized, cer_score, _, cat in stt_results:
        reply = llm_respond(recognized, model, tokenizer)
        ok, issues, bonus = check_quality(reply, cat)
        mark = f"{GREEN}✓{RESET}" if ok else f"{YELLOW}△{RESET}"
        label = f"[{cat}]"
        print(f"  {label:20s} 入力: 「{recognized[:35]}{'...' if len(recognized)>35 else ''}」")
        print(f"               応答: 「{reply}」")
        print(f"               判定: {mark}{bonus}" + (f"  [{', '.join(issues)}]" if issues else "") + "\n")
        llm_results.append(ok)

    llm_pass = sum(llm_results) / len(llm_results)
    print(f"{BOLD}Phase 2: 合格率 {llm_pass:.0%}{RESET}\n")

    # ── Phase 3: 多ターン会話テスト ──────────────────────────────
    print(f"{CYAN}{BOLD}Phase 3: 多ターン会話テスト（感情の流れ・ユーモア→深堀り）{RESET}\n")
    from app import SYSTEM_PROMPT
    convo_script = [
        ("もし世界中の誰とでも晩ごはん食べられるなら、誰と食べたい？",   "opener"),
        ("うーん、おじいちゃんかな。もう亡くなっちゃったんだけど。",      "emotional_heavy"),
        ("一緒に釣りに行ったりとか、色々教えてもらったりとかしてたんだよね。", "fond_memory"),
        ("ピザとかかな笑。冗談冗談。",                                 "humor"),
        ("一番の思い出は海で夕日を見たこと。",                          "deep"),
    ]
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    convo_ok = []
    for i, (utt, cat) in enumerate(convo_script):
        audio = await tts_to_wav(utt)
        recog = whisper_transcribe(audio)
        reply = llm_respond(recog, model, tokenizer, history)
        history.append({"role": "user",      "content": recog})
        history.append({"role": "assistant", "content": reply})
        ok, issues, bonus = check_quality(reply, cat)
        convo_ok.append(ok)
        print(f"  [{i+1}] {cat}")
        print(f"       ユーザー: {utt}")
        print(f"       認識:     {recog}")
        print(f"       AI応答:   {reply}")
        mark = f"{GREEN}✓{RESET}" if ok else f"{YELLOW}△{RESET}"
        print(f"       判定: {mark}{bonus}" + (f"  [{', '.join(issues)}]" if issues else "") + "\n")

    convo_pass = sum(convo_ok) / len(convo_ok)
    print(f"{BOLD}Phase 3: 合格率 {convo_pass:.0%}{RESET}\n")

    # ── 総合評価 ─────────────────────────────────────────────────
    print(f"{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  総合評価{RESET}")
    print(f"{'='*58}")
    print(f"  STT精度  (平均CER) : {avg_cer:.1%}   {'✓ 良好' if avg_cer < 0.15 else '△ 要改善'}")
    print(f"  LLM品質  (合格率)  : {llm_pass:.0%}    {'✓ 良好' if llm_pass >= 0.8 else '△ 要改善'}")
    print(f"  会話継続 (合格率)  : {convo_pass:.0%}    {'✓ 良好' if convo_pass >= 0.8 else '△ 要改善'}")
    print()

    suggestions = []
    if avg_cer > 0.15:
        suggestions.append("→ Whisper: initial_prompt を調整")
    if llm_pass < 0.8:
        suggestions.append("→ System prompt: ユーモア・問いかけルールを強化")
        suggestions.append("→ max_tokens を80に削減")
    if convo_pass < 0.8:
        suggestions.append("→ 会話履歴の扱いを確認")

    if not suggestions:
        print(f"  {GREEN}{BOLD}✓ 全項目良好！プロダクション品質です。{RESET}")
    else:
        print(f"  {YELLOW}改善提案:{RESET}")
        for s in suggestions:
            print(f"    {s}")

    print(f"\n{'='*58}\n")
    return avg_cer, llm_pass, convo_pass

if __name__ == "__main__":
    asyncio.run(main())
