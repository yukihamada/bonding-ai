#!/usr/bin/env python3
"""
Moshi モード end-to-end 自動テスト
ブラウザの代わりに WebSocket クライアントとして接続し、
TTS生成した音声を送信 → AI応答（音声+テキスト）を受信できるか検証
"""
import asyncio, os, sys, tempfile, time
import numpy as np
import soundfile as sf
import edge_tts
import websockets

sys.path.insert(0, os.path.dirname(__file__))

SERVER_WS = "ws://localhost:8765/ws"
TEST_UTTERANCES = [
    "こんにちは、今日はいい天気だね。",
    "うーん、好きな食べ物はラーメンかな。",
    "おじいちゃんに会いたいな。",
]

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN  = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"

async def tts_to_pcm16k(text: str) -> np.ndarray:
    """edge-tts で音声生成 → 16kHz Float32 PCM"""
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

async def test_moshi():
    print(f"\n{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  Moshi モード end-to-end テスト{RESET}")
    print(f"{BOLD}{'='*58}{RESET}\n")

    # 1) モード切替
    import aiohttp
    async with aiohttp.ClientSession() as s:
        async with s.post("http://localhost:8765/api/mode",
                          json={"mode": "moshi"}) as r:
            mode_resp = await r.json()
            print(f"{CYAN}モード切替:{RESET} {mode_resp}")

    # 2) WebSocket接続
    print(f"{CYAN}WebSocket 接続中...{RESET} {SERVER_WS}")
    ws = await websockets.connect(SERVER_WS, max_size=2**24)
    print(f"{GREEN}✓ 接続完了{RESET}\n")

    # 3) 受信タスク起動
    received_text = []
    received_audio_bytes = 0
    received_greeting = None
    stop_event = asyncio.Event()
    moshi_ready = asyncio.Event()

    async def receiver():
        nonlocal received_audio_bytes, received_greeting
        try:
            while not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(msg, bytes):
                    received_audio_bytes += len(msg)
                else:
                    import json
                    data = json.loads(msg)
                    kind = data.get("type")
                    text = data.get("text", "")
                    if kind == "ai":
                        if received_greeting is None:
                            received_greeting = text
                            print(f"{GREEN}✓ AI挨拶受信:{RESET} 「{text[:60]}...」")
                        else:
                            received_text.append(text)
                            print(f"{GREEN}→ AI返答:{RESET} 「{text[:60]}」")
                    elif kind == "status":
                        print(f"{YELLOW}[status]{RESET} {text}")
                        if "準備完了" in text:
                            moshi_ready.set()
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            print(f"{RED}receiver error:{RESET} {e}")

    recv_task = asyncio.create_task(receiver())

    # 4) Moshi準備完了を待つ（最大5分）
    print(f"{CYAN}Moshi 起動待ち（最大5分）...{RESET}")
    try:
        await asyncio.wait_for(moshi_ready.wait(), timeout=300)
        print(f"{GREEN}✓ Moshi 準備完了{RESET}\n")
    except asyncio.TimeoutError:
        print(f"{RED}✗ Moshi 準備タイムアウト（300秒）{RESET}")
        stop_event.set()
        await recv_task
        await ws.close()
        return False

    # 5) TTS音声を送信
    total_sent = 0
    for i, utt in enumerate(TEST_UTTERANCES):
        print(f"{CYAN}発話 {i+1}/{len(TEST_UTTERANCES)}:{RESET} 「{utt}」")
        pcm = await tts_to_pcm16k(utt)
        # 4096サンプルずつ送信（ブラウザの scriptProcessor サイズを模倣）
        CHUNK = 4096
        for j in range(0, len(pcm), CHUNK):
            chunk = pcm[j:j+CHUNK]
            await ws.send(chunk.tobytes())
            total_sent += len(chunk)
            await asyncio.sleep(CHUNK / 16000.0)  # リアルタイム速度
        # Moshi が応答するまで少し待つ
        await asyncio.sleep(3)

    # 最終応答を待つ
    print(f"\n{CYAN}応答待機中（10秒）...{RESET}")
    await asyncio.sleep(10)
    stop_event.set()
    await recv_task
    await ws.close()

    # 6) 結果判定
    print(f"\n{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  結果{RESET}")
    print(f"{'='*58}")
    checks = [
        ("AI挨拶受信",       received_greeting is not None),
        ("Moshi準備完了",    moshi_ready.is_set()),
        ("音声データ受信",   received_audio_bytes > 1000),
        ("送信サンプル数",   total_sent > 16000),
    ]
    all_ok = True
    for name, ok in checks:
        mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {mark} {name}")
        if not ok: all_ok = False
    print(f"  送信合計: {total_sent:,} samples ({total_sent/16000:.1f}秒)")
    print(f"  受信音声: {received_audio_bytes:,} bytes")
    print(f"  AI発言数: 挨拶1 + 応答{len(received_text)}")
    print()
    if all_ok:
        print(f"  {GREEN}{BOLD}✓ Moshi モード動作OK{RESET}")
    else:
        print(f"  {RED}{BOLD}✗ Moshi モードに問題あり{RESET}")
    print(f"{'='*58}\n")
    return all_ok

if __name__ == "__main__":
    ok = asyncio.run(test_moshi())
    sys.exit(0 if ok else 1)
