"""
omnivoice_bridge.py — Cầu TTS cho dự án OmniVoice.

Sinh giọng Ngọc Huyền bằng model FINE-TUNE VieNeu-TTS-0.3B (LoRA đã merge, cục bộ),
chạy bằng .venv RIÊNG của VieNeu-TTS để không đụng stack phụ thuộc của OmniVoice.

Cách gọi (từ GUI OmniVoice):
    .venv/Scripts/python.exe omnivoice_bridge.py --text-file in.txt --output out.wav

Nạp THẲNG merged cache (standard backend, KHÔNG LMDeploy/triton — vốn hay lỗi trên
Windows) → dùng giọng mặc định "NgocHuyen" trong voices.json. Trả về wav 24kHz.
Thoát mã 0 nếu OK; 2 nếu văn bản rỗng; 1 nếu lỗi khác.
"""
import argparse
import re
import sys
from pathlib import Path

# Console Windows mặc định cp1252 → in tiếng Việt / emoji (kể cả print trong
# vieneu/core.py) sẽ UnicodeEncodeError. Ép UTF-8 cho mọi output của tiến trình.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "merged_models_cache" / "pnnbao-ump_VieNeu-TTS-0.3B-lora-ngoc-huyen"
DEFAULT_CODEC = "neuphonic/distill-neucodec"


def log(msg: str) -> None:
    print(f"[bridge] {msg}", flush=True)


def sanitize_input_text(text: str) -> str:
    """Làm sạch tối thiểu để phoneme hoá ổn định (giống text_to_voice_ngoc_huyen.py):
    bỏ ký tự vô hình, bỏ hệ chữ KHÔNG Latin, giữ chữ Việt + dấu câu thường dùng."""
    value = (text or "").strip()
    if not value:
        return ""
    value = re.sub(r"[​‌‍﻿]", "", value)
    non_latin = r"[Ѐ-ӿ֐-׿؀-ۿ一-鿿぀-ヿ가-힯]"
    value = re.sub(non_latin, " ", value)
    value = re.sub(r"[^0-9A-Za-zÀ-ỹĐđ\s\.,!?;:'\"()\-\n/%&+=]", " ", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return value


def apply_guardrails(text: str, temperature: float, max_chars: int, top_k: int):
    """Văn bản dài dễ 'đọc lạc' → hạ temperature/max_chars/top_k cho ổn định."""
    t, mc, tk = float(temperature), int(max_chars), int(top_k)
    if len(text) > 260:
        t = min(t, 0.8)
        mc = min(mc, 256)
        tk = min(tk, 35)
    mc = max(128, min(1024, mc))
    tk = max(1, min(100, tk))
    t = max(0.1, min(1.5, t))
    return t, mc, tk


def detect_device(dev: str) -> str:
    import torch
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and not torch.cuda.is_available():
        log("CUDA không khả dụng → lùi về CPU.")
        return "cpu"
    return dev


def main() -> None:
    p = argparse.ArgumentParser(description="Cầu VieNeu-TTS (giọng Ngọc Huyền) cho OmniVoice.")
    p.add_argument("--text-file", required=True, help="File .txt đầu vào (UTF-8).")
    p.add_argument("--output", required=True, help="Đường dẫn wav đầu ra.")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="Thư mục merged model.")
    p.add_argument("--codec", default=DEFAULT_CODEC, help="Repo codec.")
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-chars", type=int, default=256)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--silence-p", type=float, default=0.08)
    args = p.parse_args()

    raw = Path(args.text_file).read_text(encoding="utf-8")
    text = sanitize_input_text(raw)
    if not text:
        log("LỖI: văn bản rỗng sau khi làm sạch.")
        sys.exit(2)

    model_path = args.model
    if not Path(model_path).exists():
        log(f"LỖI: không thấy thư mục model: {model_path}")
        sys.exit(1)

    device = detect_device(args.device)
    temp, max_chars, top_k = apply_guardrails(text, args.temperature, args.max_chars, args.top_k)
    log(f"device={device} | chars={len(text)} | temp={temp} max_chars={max_chars} top_k={top_k}")

    from vieneu import VieNeuTTS

    log(f"Nạp model (standard): {model_path}")
    tts = VieNeuTTS(
        backbone_repo=model_path,
        backbone_device=device,
        codec_repo=args.codec,
        codec_device=device,
    )
    try:
        log("Đang sinh giọng...")
        audio = tts.infer(
            text=text,
            temperature=temp,
            max_chars=max_chars,
            top_k=top_k,
            silence_p=args.silence_p,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        tts.save(audio, str(out))
        log(f"Xong → {out}")
    finally:
        try:
            tts.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
