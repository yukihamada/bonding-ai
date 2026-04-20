#!/usr/bin/env python3
"""
Pipeline モード end-to-end テスト
ブラウザと同じ WebSocket 経由で音声を送信し、STT→LLM→応答まで動くか検証
"""
import asyncio, json, os, sys, tempfile, time
import numpy as np
import soundfile as sf
import edge_tts
import websockets
import aiohttp

SERVER_WS   = "ws://localhost:8765/ws"
SERVER_HTTP = "http://localhost:8765"

TEST_UTTERANCES = [
    "好きな食べ物はラーメンかな。",
    "うーん、おじいちゃんかな。もう亡くなっちゃったんだけど。",
]

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN  = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"

async def tts_to_pcm16k(text: str) -> np.ndarray:
    c = edge_tts.Communicate(text, voice="ja-JP-NanamiNeural", rate="+5%")
    data = b""
    async for chunk in c.stream():
        if chunk["type"] == "audio": data += chunk["data"]
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(data); fname = f.name
    audio, sr = sf.read(fname)
    os.unlink(fname)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = np.gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g)
    return audio.astype(np.float32)

async def test_pipeline():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Pipeline モード end-to-end テスト{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    # 1) モード切替
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{SERVER_HTTP}/api/mode", json={"mode": "pipeline"}) as r:
            print(f"{CYAN}モード切替:{RESET} {await r.json()}")

    # 2) WebSocket接続
    ws = await websockets.connect(SERVER_WS, max_size=2**24, ping_interval=None)
    print(f"{GREEN}✓ WebSocket接続{RESET}\n")

    greeting = None
    user_texts = []
    ai_replies = []
    status_msgs = []
    stop = asyncio.Event()
    speaking_done = asyncio.Event()
    ai_is_speaking = {"on": False}

    async def receiver():
        nonlocal greeting
        try:
            while not stop.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(msg, str):
                    d = json.loads(msg)
                    t, text = d.get("type"), d.get("text", "")
                    if t == "ai":
                        if greeting is None:
                            greeting = text
                            print(f"{GREEN}[AI挨拶]{RESET} {text[:80]}...")
                        else:
                            ai_replies.append(text)
                            print(f"{GREEN}[AI応答]{RESET} {text}")
                    elif t == "user":
                        user_texts.append(text)
                        print(f"{CYAN}[STT認識]{RESET} {text}")
                    elif t == "status":
                        status_msgs.append(text)
                    elif t == "speaking":
                        on = d.get("on", False)
                        ai_is_speaking["on"] = on
                        if not on:
                            speaking_done.set()
        except websockets.ConnectionClosed:
            pass

    recv_task = asyncio.create_task(receiver())

    # 3) 挨拶再生終了まで待つ（_speaking=True の間は入力が捨てられる）
    print(f"{YELLOW}挨拶再生終了待ち...{RESET}")
    try:
        await asyncio.wait_for(speaking_done.wait(), timeout=30)
        print(f"{GREEN}✓ 挨拶再生終了{RESET}")
    except asyncio.TimeoutError:
        print(f"{YELLOW}挨拶再生タイムアウト（続行）{RESET}")
    await asyncio.sleep(0.5)

    # 4) 音声送信（発話ごと）
    CHUNK = 4096
    response_times = []
    for i, utt in enumerate(TEST_UTTERANCES):
        # AIが喋ってる間は待つ
        while ai_is_speaking["on"]:
            await asyncio.sleep(0.1)

        print(f"\n{CYAN}▶ 発話 {i+1}/{len(TEST_UTTERANCES)}:{RESET} 「{utt}」")
        pcm = await tts_to_pcm16k(utt)
        t0 = time.time()
        for j in range(0, len(pcm), CHUNK):
            chunk = pcm[j:j+CHUNK]
            await ws.send(chunk.tobytes())
            await asyncio.sleep(CHUNK / 16000.0)
        # 無音を1.0秒送ってVADトリガ
        silence = np.zeros(int(16000 * 1.0), dtype=np.float32)
        for j in range(0, len(silence), CHUNK):
            await ws.send(silence[j:j+CHUNK].tobytes())
            await asyncio.sleep(CHUNK / 16000.0)
        # 応答を90秒まで待つ（mlx JIT初回コンパイルで最大20秒かかるため）
        target_replies = i + 1
        for _ in range(180):
            if len(ai_replies) >= target_replies:
                response_times.append(time.time() - t0)
                break
            await asyncio.sleep(0.5)

    stop.set()
    await recv_task
    await ws.close()

    # 5) 判定
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  結果{RESET}")
    print(f"{'='*60}")
    import re
    japanese_ok   = all(bool(re.search(r'[ぁ-んァ-ン]', r)) for r in ai_replies) if ai_replies else False
    question_ok   = all((r.endswith('？') or r.endswith('?') or '？' in r[-5:]) for r in ai_replies) if ai_replies else False
    stt_ok        = len(user_texts) >= len(TEST_UTTERANCES)
    replies_ok    = len(ai_replies) >= len(TEST_UTTERANCES)

    checks = [
        ("挨拶受信",           greeting is not None),
        ("STT認識（全発話）",  stt_ok),
        ("AI応答（全発話）",   replies_ok),
        ("応答が日本語",       japanese_ok),
        ("応答が問いかけ",     question_ok),
    ]
    all_ok = True
    for name, ok in checks:
        mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {mark} {name}")
        if not ok: all_ok = False

    print(f"  挨拶: {greeting[:60] if greeting else 'None'}...")
    print(f"  STT結果 ({len(user_texts)}): {user_texts}")
    print(f"  AI応答 ({len(ai_replies)}): {[r[:40]+'...' if len(r)>40 else r for r in ai_replies]}")
    if response_times:
        print(f"  {BOLD}応答時間 (発話終了→AI応答):{RESET} {[f'{t:.1f}秒' for t in response_times]}")
        print(f"  {BOLD}平均応答時間:{RESET} {sum(response_times)/len(response_times):.1f}秒")
    print()
    if all_ok:
        print(f"  {GREEN}{BOLD}✓ Pipeline モード end-to-end OK{RESET}")
    else:
        print(f"  {RED}{BOLD}✗ Pipeline モードに問題あり{RESET}")
    print(f"{'='*60}\n")
    return all_ok

if __name__ == "__main__":
    ok = asyncio.run(test_pipeline())
    sys.exit(0 if ok else 1)
