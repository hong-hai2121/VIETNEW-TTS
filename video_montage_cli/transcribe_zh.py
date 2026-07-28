#!/usr/bin/env python3
# Installation:
#   pip install faster-whisper
# Example:
#   python transcribe_zh.py --input "sample.mp3"

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "medium"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_MODEL_CACHE_DIR = SCRIPT_DIR / "models" / "faster_whisper"


def detect_device_config() -> List[Dict[str, str]]:
    """Return ordered device/compute configs with GPU-first fallback."""
    gpu_available = False

    if shutil.which("nvidia-smi"):
        try:
            probe = subprocess.run(
                ["nvidia-smi", "-L"],
                check=False,
                capture_output=True,
                text=True,
            )
            gpu_available = probe.returncode == 0
        except Exception:
            gpu_available = False

    # Optional torch check if available.
    if not gpu_available:
        try:
            import torch  # type: ignore

            gpu_available = bool(torch.cuda.is_available())
        except Exception:
            gpu_available = False

    if gpu_available:
        return [
            {"device": "cuda", "compute_type": "float16"},
            {"device": "cuda", "compute_type": "int8_float16"},
            {"device": "cuda", "compute_type": "int8"},
            {"device": "cpu", "compute_type": "int8"},
        ]

    return [{"device": "cpu", "compute_type": "int8"}]


def load_model(model_name: str, model_cache_dir: Path) -> Tuple["WhisperModel", Dict[str, str]]:
    """Load Whisper model with automatic fallback across compute configs."""
    if WhisperModel is None:
        raise RuntimeError(
            "Missing dependency 'faster-whisper'. Install with: pip install faster-whisper"
        )

    model_cache_dir.mkdir(parents=True, exist_ok=True)
    configs = detect_device_config()
    errors: List[str] = []

    for cfg in configs:
        device = cfg["device"]
        compute_type = cfg["compute_type"]
        try:
            print(
                f"[INFO] Loading model='{model_name}' "
                f"(device={device}, compute_type={compute_type})"
            )
            model = WhisperModel(
                model_size_or_path=model_name,
                device=device,
                compute_type=compute_type,
                download_root=str(model_cache_dir.resolve()),
            )
            return model, cfg
        except Exception as err:
            errors.append(f"{device}/{compute_type}: {err}")
            print(f"[WARN] Failed to load with {device}/{compute_type}: {err}")

    joined = "\n".join(errors)
    raise RuntimeError(f"Unable to load model '{model_name}'. Attempts:\n{joined}")


def transcribe_file(
    model: "WhisperModel",
    input_path: Path,
    beam_size: int,
) -> List[Dict[str, Any]]:
    """Transcribe input media into Chinese text segments."""
    try:
        segments, _ = model.transcribe(
            str(input_path),
            language="zh",
            task="transcribe",
            beam_size=beam_size,
        )
    except Exception as err:
        message = str(err).lower()
        if "ffmpeg" in message or "avformat" in message or "error opening input" in message:
            raise RuntimeError(
                "Cannot read media file. Please install FFmpeg and ensure 'ffmpeg' is in PATH."
            ) from err
        raise

    results: List[Dict[str, Any]] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        results.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
            }
        )
    return results


def save_txt(output_path: Path, segments: List[Dict[str, Any]]) -> None:
    """Save plain text transcript without timestamps."""
    lines = [str(seg["text"]).strip() for seg in segments if str(seg["text"]).strip()]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def save_srt(output_path: Path, segments: List[Dict[str, Any]]) -> None:
    """Save subtitle file in standard SRT format."""
    lines: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        start_ts = format_srt_timestamp(float(seg["start"]))
        end_ts = format_srt_timestamp(float(seg["end"]))
        text = str(seg["text"]).strip()
        lines.extend([str(idx), f"{start_ts} --> {end_ts}", text, ""])
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def save_json(output_path: Path, segments: List[Dict[str, Any]]) -> None:
    """Save segment metadata to JSON."""
    output_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")


def choose_input_file() -> Optional[Path]:
    """Pick input media path from file picker or console when --input is not provided."""
    if tk is not None and filedialog is not None:
        root = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.update_idletasks()
            selected = filedialog.askopenfilename(
                title="Select input audio/video file",
                filetypes=[
                    ("Media files", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.mp4 *.mkv *.mov *.webm *.avi"),
                    ("All files", "*.*"),
                ],
            )
            if selected:
                return Path(selected)
        except Exception:
            pass
        finally:
            if root is not None:
                root.destroy()

    try:
        raw_path = input("Nhap duong dan file audio/video (--input): ").strip().strip('"')
    except EOFError:
        return None
    if not raw_path:
        return None
    return Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe Chinese speech/audio from media using faster-whisper."
    )
    parser.add_argument("--input", required=False, type=Path, help="Path to input audio/video file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model name (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.as_posix()})",
    )
    parser.add_argument("--beam_size", default=5, type=int, help="Beam size for decoding (default: 5)")
    args = parser.parse_args()

    input_candidate = args.input
    if input_candidate is None:
        input_candidate = choose_input_file()
        if input_candidate is None:
            print("[ERROR] Missing input file. Use --input or choose a file when prompted.", file=sys.stderr)
            return 1

    input_path = input_candidate.expanduser().resolve()
    if not input_path.exists() or not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem
    txt_path = output_dir / f"{base_name}.txt"
    srt_path = output_dir / f"{base_name}.srt"
    json_path = output_dir / f"{base_name}.json"

    print(f"[INFO] Input file: {input_path}")
    print(f"[INFO] Model name: {args.model}")
    print(f"[INFO] Model cache path: {DEFAULT_MODEL_CACHE_DIR.resolve()}")
    print(f"[INFO] Output files:")
    print(f"       - {txt_path}")
    print(f"       - {srt_path}")
    print(f"       - {json_path}")

    try:
        model, runtime_cfg = load_model(args.model, DEFAULT_MODEL_CACHE_DIR)
        runtime_device = runtime_cfg["device"]
        runtime_compute = runtime_cfg["compute_type"]
        print(f"[INFO] Runtime: device={runtime_device}, compute_type={runtime_compute}")
        print(f"[INFO] Using {'GPU (NVIDIA CUDA)' if runtime_device == 'cuda' else 'CPU'}")

        segments = transcribe_file(model, input_path, args.beam_size)
        save_txt(txt_path, segments)
        save_srt(srt_path, segments)
        save_json(json_path, segments)
    except RuntimeError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"[ERROR] Unexpected error: {err}", file=sys.stderr)
        return 1

    print("[DONE] Transcription completed successfully.")
    print(f"[DONE] TXT : {txt_path}")
    print(f"[DONE] SRT : {srt_path}")
    print(f"[DONE] JSON: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
