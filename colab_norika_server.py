#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Colab NVIDIA T4 GPU 音声推論サーバー (Style-Bert-VITS2 藤原紀香公式モデル)
- 1セル実行で全自動セットアップ (GPU検証 -> モデル自動取得 -> サーバー起動 -> トンネル開通 -> 自宅PC自動登録)
- 音声推論速度: 約 0.2〜0.3 秒 (手元 CPU の 25秒から 80倍高速化)
"""

import os
import sys
import time
import json
import re
import shutil
import subprocess
import threading
from pathlib import Path
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# 自宅 PC の公開エンドポイント (固定ドメイン)
HOME_TUNNEL_URL = os.environ.get("HOME_VOICE_URL", "https://voice.moshaj.com")

# 作業ディレクトリ
BASE_DIR = Path("/content/Style-Bert-VITS2") if Path("/content/Style-Bert-VITS2").exists() else (Path("/content/norika_gpu_server") if Path("/content").exists() else Path("./norika_gpu_server"))
if Path("/content/Style-Bert-VITS2").exists():
    sys.path.insert(0, "/content/Style-Bert-VITS2")

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "Norika_Official_e120_s480.safetensors"
CONFIG_PATH = MODEL_DIR / "config.json"
STYLE_PATH = MODEL_DIR / "style_vectors.npy"

PORT = 50050
_tts_model = None

def check_gpu():
    print("=======================================================", flush=True)
    print("  [1/5] GPU 稼働状態の事前健全性チェック (Pre-flight Check)", flush=True)
    print("=======================================================", flush=True)
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n❌ [ERROR] NVIDIA GPU が検出されませんでした！", flush=True)
            print("[INFO] Colab メニューの「ランタイム」>「ランタイムのタイプを変更」から「T4 GPU」を選択してください。\n", flush=True)
            sys.exit(1)
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU 検出成功: {device_name} (VRAM: {vram_gb:.1f} GB)", flush=True)
        print(f"✅ PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
    except Exception as ex:
        print(f"❌ GPU チェック失敗: {ex}", flush=True)
        sys.exit(1)

def install_cloudflared():
    print("\n[2/5] Cloudflare Tunnel (cloudflared) の準備...", flush=True)
    cf_bin = Path("/usr/local/bin/cloudflared")
    if not cf_bin.exists():
        print("  cloudflared をダウンロード中...", flush=True)
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        try:
            urllib.request.urlretrieve(url, str(cf_bin))
            os.chmod(str(cf_bin), 0o755)
            print("  ✅ cloudflared インストール完了", flush=True)
        except Exception as ex:
            print(f"  ❌ cloudflared ダウンロード失敗: {ex}", flush=True)
    else:
        print("  ✅ cloudflared 準備済み", flush=True)

RELEASE_BASE_URL = "https://github.com/moshajmoshaj/norika-voice-dataset/releases/download/v2.0.0"
HOME_MODEL_FALLBACK_URL = os.environ.get("HOME_VOICE_URL", "https://voice.moshaj.com")

def download_model():
    print("\n[3/5] 高速 CDN (GitHub Releases) から藤原紀香 AIボイス公式モデルを瞬速取得...", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        ("config.json", CONFIG_PATH),
        ("style_vectors.npy", STYLE_PATH),
        ("Norika_Official_e120_s480.safetensors", MODEL_PATH)
    ]

    for fname, target_path in files:
        if target_path.exists() and target_path.stat().st_size > 1000:
            print(f"  [OK] 既存モデルあり: {fname} ({target_path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
            continue

        cdn_url = f"{RELEASE_BASE_URL}/{fname}"
        print(f"  [FAST-CDN] 高速ダウンロード中: {fname}...", flush=True)
        try:
            subprocess.run(["curl", "-LsSf", "-o", str(target_path), cdn_url], check=True)
            print(f"  [OK] 取得完了: {fname} ({target_path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
        except Exception as ex:
            print(f"  [WARN] CDN 取得失敗 ({fname}): {ex} -> 自宅 PC からフォールバック取得中...", flush=True)
            fallback_url = f"{HOME_MODEL_FALLBACK_URL}/model/{fname}"
            subprocess.run(["curl", "-LsSf", "-o", str(target_path), fallback_url], check=True)

def load_vits2_model():
    global _tts_model
    print("\n[4/5] NVIDIA T4 GPU に Style-Bert-VITS2 モデルを一括ロード中...", flush=True)
    t0 = time.time()
    from style_bert_vits2.tts_model import TTSModel

    _tts_model = TTSModel(
        model_path=Path(MODEL_PATH),
        config_path=Path(CONFIG_PATH),
        style_vec_path=Path(STYLE_PATH),
        device="cuda"
    )
    load_time = time.time() - t0
    print(f"✅ 藤原紀香公式モデル GPU ロード完了 ({load_time:.2f} 秒)！", flush=True)

    # 初回ウォームアップ推論
    print("  初回ウォームアップ推論中...", flush=True)
    tw0 = time.time()
    sr, _ = _tts_model.infer(text="こんにちは。ふじわらのりかです。")
    print(f"  ✅ ウォームアップ完了 ({time.time() - tw0:.2f} 秒, サンプリングレート: {sr}Hz)", flush=True)

def normalize_text(text: str) -> str:
    t = text.replace("！", "。").replace("!", "。").replace("？", "。").replace("?", "。")
    t = t.replace("～", "ー").replace("〜", "ー").replace("・", "、")
    t = re.sub(r"[^\w\s、。ー「」『』\(\)（）]", "", t)
    t = re.sub(r"([。、])\1+", r"\1", t)
    t = t.replace("藤原紀香", "ふじわらのりか")
    if not t:
        t = "はい。"
    elif not t.endswith("。") and not t.endswith("、"):
        t += "。"
    return t

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class NorikaGpuHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            import torch
            res = {
                "status": "ok",
                "engine": "Style-Bert-VITS2-GPU",
                "device": "cuda",
                "gpu": torch.cuda.get_device_name(0),
                "model": MODEL_PATH.name
            }
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            return

        if parsed.path in ["/voice", "/synthesis"]:
            qs = urllib.parse.parse_qs(parsed.query, encoding="utf-8")
            text = qs.get("text", [""])[0]
            if not text:
                self.send_error(400, "text is required")
                return

            try:
                norm_text = normalize_text(text)
                t0 = time.time()
                sr, audio_array = _tts_model.infer(text=norm_text)
                infer_time = time.time() - t0

                # WAV バイト列にエンコード
                import io
                import soundfile as sf
                wav_io = io.BytesIO()
                sf.write(wav_io, audio_array, sr, format="WAV", subtype="PCM_16")
                wav_bytes = wav_io.getvalue()

                print(f"[GPU INFER SUCCESS: {infer_time*1000:.1f}ms] {len(wav_bytes)/1024:.1f}KB for: 「{norm_text}」", flush=True)

                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("X-Inference-Time-Ms", f"{infer_time*1000:.1f}")
                self.end_headers()
                self.wfile.write(wav_bytes)
            except Exception as ex:
                print(f"[GPU INFER ERROR] {ex}", flush=True)
                self.send_error(500, str(ex))
            return

        self.send_error(404, "Not Found")

def start_server_and_tunnel():
    print("\n[5/5] HTTP 推論サーバー起動 ＆ Cloudflare Tunnel 自動接続...", flush=True)

    # 1. ローカル HTTP サーバーをバックグラウンド起動
    server = ThreadingHTTPServer(("127.0.0.1", PORT), NorikaGpuHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"  ✅ ローカル推論サーバー稼働中: http://127.0.0.1:{PORT}", flush=True)

    # 2. Cloudflare Tunnel 起動
    cmd = ["/usr/local/bin/cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    tunnel_url = None
    url_regex = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    for _ in range(40):
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.5)
            continue
        match = url_regex.search(line)
        if match:
            tunnel_url = match.group(0)
            break

    if not tunnel_url:
        print("❌ Cloudflare Tunnel URL の取得に失敗しました。再試行してください。", flush=True)
        return

    print(f"\n=======================================================", flush=True)
    print(f"  [OK] 祝・開通！藤原紀香 AIボイス Google Colab GPU サーバー", flush=True)
    print(f"  [LINK] Colab Tunnel URL: {tunnel_url}", flush=True)
    print(f"=======================================================", flush=True)

    # 3. 自宅 PC の HomeVoiceHub へ自動登録
    print(f"\n  自宅 PC ({HOME_TUNNEL_URL}) へ Colab GPU を自動登録中...", flush=True)
    try:
        reg_url = f"{HOME_TUNNEL_URL}/api/colab/register"
        req_data = json.dumps({"url": tunnel_url}).encode("utf-8")
        req = urllib.request.Request(
            reg_url,
            data=req_data,
            headers={"Content-Type": "application/json", "User-Agent": "ColabNorikaGpu/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_body = resp.read().decode("utf-8")
            print(f"  ✅ 自宅 PC への自動登録完了！: {resp_body}", flush=True)
    except Exception as ex:
        print(f"  ⚠️ 自宅 PC への自動通知でエラー (手動でも可): {ex}", flush=True)

    print("\n[Norika] すべての準備が完了しました！", flush=True)
    print("[INFO] スマホから話しかけると、NVIDIA GPU により【0.3秒】で紀香ボイスが即座に返ってきます！", flush=True)
    print("（このセルの実行を停止するまで、GPU推論サーバーは常駐し続けます）\n", flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nサーバーを停止しました。")
        proc.terminate()
        server.shutdown()

if __name__ == "__main__":
    check_gpu()
    install_cloudflared()
    download_model()
    load_vits2_model()
    start_server_and_tunnel()
