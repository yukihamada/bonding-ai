#!/usr/bin/env python3
"""
Moshi 認識・応答品質テスト（単一セッション・8発話連続）
"""
import asyncio, json, os, re, sys, tempfile, time
import numpy as np
import soundfile as sf
import edge_tts
import websockets
import aiohttp

SERVER_WS   = "ws://localhost:8765/ws"
SERVER_HTTP = "http://localhost:8765"

TEST_CASES = [
    ("greeting",  "こんにちは、今日はいい天気だね。"),
    ("food",      "お昼ご飯、何食べた？"),
    ("food2",     "ラーメン好きなんだよね。"),
    ("emotion",   "最近ちょっと疲れてて、休みたい。"),
    ("question",  "好きな映画は何？"),
    ("memory",    "子どもの頃、夏休みに田舎に行ってた。"),
    ("short",     "うん、そうだね。"),
    ("deep",      "大切な人を失うのって、どう乗り越えるの？"),
]

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN  = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"

async def tts(text):
    c = edge_tts.Communicate(text, voice="ja-JP-NanamiNeural", rate="+5%")
    data = b""
    async for chunk in c.stream():
        if chunk["type"] == "audio": data += chunk["data"]
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(data); fname = f.name
    audio, sr = sf.read(fname); os.unlink(fname)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = np.gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g)
    return audio.astype(np.float32)

def score(user_text, moshi_text):
    if not moshi_text: return 0, "応答なし"
    issues = []
    s = 0
    if re.search(r'[ぁ-んァ-ン]', moshi_text): s += 2
    else: issues.append("日本語なし")
    if 2 <= len(moshi_text) <= 150: s += 1
    elif len(moshi_text) < 2: issues.append("短すぎ")
    else: issues.append(f"長すぎ({len(moshi_text)})")
    if len(set(moshi_text)) / max(len(moshi_text), 1) >= 0.2: s += 1
    else: issues.append("繰り返し多い")
    if any(len(w) >= 2 for w in re.split(r'[、。！？\s]', moshi_text)): s += 1
    return s, ", ".join(issues) if issues else "OK"

async def main():
    print(f"\n{BOLD}{'='*62}{RESET}")
    print(f"{BOLD}  Moshi 認識・応答品質テスト（単一セッション）{RESET}")
    print(f"{BOLD}{'='*62}{RESET}")

    async with aiohttp.ClientSession() as s:
        await s.post(f"{SERVER_HTTP}/api/mode", json={"mode": "moshi"})

    ws = await websockets.connect(SERVER_WS, max_size=2**24)
    print(f"{GREEN}✓ 接続{RESET}\n")

    greeting_text = None
    moshi_ready = asyncio.Event()
    # 発話ごとに区切るため、現在の発話index を共有
    current_case = {"idx": -1, "start_time": None}
    case_audio_bytes = [0] * len(TEST_CASES)
    case_replies    = [[] for _ in TEST_CASES]
    stop = asyncio.Event()

    async def receiver():
        nonlocal greeting_text
        try:
            while not stop.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                idx = current_case["idx"]
                if isinstance(msg, bytes):
                    if idx >= 0:
                        case_audio_bytes[idx] += len(msg)
                else:
                    d = json.loads(msg)
                    t = d.get("type"); text = d.get("text", "")
                    if t == "ai":
                        if greeting_text is None:
                            greeting_text = text
                            print(f"{GREEN}[挨拶]{RESET} {text[:70]}{'...' if len(text)>70 else ''}")
                        elif idx >= 0:
                            case_replies[idx].append(text)
                    elif t == "status":
                        print(f"{YELLOW}[status]{RESET} {text}")
                        if "準備完了" in text: moshi_ready.set()
        except websockets.ConnectionClosed:
            pass

    recv_task = asyncio.create_task(receiver())

    # Moshi ready待ち（最大5分）
    try:
        await asyncio.wait_for(moshi_ready.wait(), timeout=300)
    except asyncio.TimeoutError:
        print(f"{RED}Moshi ready timeout{RESET}"); stop.set(); await recv_task; return False
    await asyncio.sleep(2)

    # 連続で発話送信
    CHUNK = 4096
    for i, (label, utt) in enumerate(TEST_CASES):
        current_case["idx"] = i
        current_case["start_time"] = time.time()
        print(f"\n{CYAN}▶ [{label}]{RESET} 入力: 「{utt}」")
        pcm = await tts(utt)
        for j in range(0, len(pcm), CHUNK):
            await ws.send(pcm[j:j+CHUNK].tobytes())
            await asyncio.sleep(CHUNK / 16000.0)
        # 発話終了後、Moshiの応答を8秒収集
        await asyncio.sleep(8)

    stop.set()
    await recv_task
    await ws.close()

    # 集計
    print(f"\n{BOLD}{'='*62}{RESET}")
    print(f"{BOLD}  結果サマリ{RESET}")
    print(f"{'='*62}")
    results = []
    for i, (label, utt) in enumerate(TEST_CASES):
        reply = "".join(case_replies[i])
        sc, note = score(utt, reply)
        results.append((label, utt, reply, case_audio_bytes[i], sc, note))

    avg_score = sum(r[4] for r in results) / len(results)
    avg_chars = sum(len(r[2]) for r in results) / len(results)
    avg_audio = sum(r[3] for r in results) / len(results)
    ja_rate   = sum(1 for r in results if re.search(r'[ぁ-んァ-ン]', r[2])) / len(results)

    print(f"  テスト数:       {len(results)}")
    print(f"  平均スコア:     {avg_score:.1f} / 5")
    print(f"  平均応答文字:   {avg_chars:.0f}")
    print(f"  平均音声:       {avg_audio/1024:.1f} KB")
    print(f"  日本語率:       {ja_rate:.0%}")
    print()
    print(f"  {BOLD}ケース別:{RESET}")
    for label, utt, reply, ab, sc, note in results:
        mark = f"{GREEN}✓{RESET}" if sc >= 4 else f"{YELLOW}△{RESET}" if sc >= 2 else f"{RED}✗{RESET}"
        print(f"    {mark} [{label:10s}] {sc}/5 音声{ab/1024:.0f}KB ─ 「{reply[:60]}{'...' if len(reply)>60 else ''}」")
        if note != "OK": print(f"                     └ {note}")

    print(f"\n{BOLD}{'='*62}{RESET}")
    ok = avg_score >= 3.0 and ja_rate >= 0.8
    print(f"  {GREEN if ok else YELLOW}{BOLD}{'✓ Moshi品質合格' if ok else '△ Moshi品質要改善'}{RESET}")
    print(f"{'='*62}\n")
    return ok

if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
