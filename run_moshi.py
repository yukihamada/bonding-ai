#!/usr/bin/env python3
"""
つながりAI — Moshi S2S版 (moshi_mlx.local)
完全ローカル・Apple Silicon MLX で動作

Moshi は純粋な S2S ニューラルモデル（システムプロンプト不可）。
英語メインの自由会話デモとして使用。

起動: python3.12 run_moshi.py
終了: Ctrl+C
"""
import subprocess, sys, os

MOSHI_REPO  = "kyutai/moshiko-mlx-q4"
PYTHON      = sys.executable

def main():
    print("━" * 42)
    print("   つながりAI — Moshi S2S版")
    print(f"   モデル: {MOSHI_REPO}")
    print("   完全ローカル | Ctrl+C で終了")
    print("━" * 42)
    print()
    print("Moshi モデル読み込み中（初回は数分かかります）...")
    print("準備完了後、話しかけてください。")
    print()

    cmd = [PYTHON, "-m", "moshi_mlx.local", "--hf-repo", MOSHI_REPO, "-q", "4"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n終了しました。")

if __name__ == "__main__":
    main()
