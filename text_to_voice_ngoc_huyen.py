import argparse
import atexit
import json
import re
import shutil
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

import gradio as gr
import numpy as np
import soundfile as sf
import torch

try:
    import librosa
except Exception:
    librosa = None

from vieneu import FastVieNeuTTS, VieNeuTTS


DEFAULT_BASE_MODEL = "pnnbao-ump/VieNeu-TTS-0.3B"
DEFAULT_LORA_MODEL = "pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen"
DEFAULT_CODEC_MODEL = "neuphonic/distill-neucodec"
MERGED_CACHE_ROOT = Path("merged_models_cache")

_MODEL_LOCK = Lock()
_MODEL_INSTANCE = None
_MODEL_SIGNATURE = None
_MODEL_BACKEND = None
_MODEL_MERGED_PATH = None
_MODEL_LAST_ERROR = None


def get_supported_devices() -> list[str]:
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def detect_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_text_input(text: str | None, text_file: str | None) -> str:
    if text:
        value = text.strip()
        if value:
            return value
    if text_file:
        value = read_text_file_content(text_file)
        if value:
            return value.strip()
    raise ValueError("Vui long cung cap --text hoac --text-file.")


def sanitize_input_text(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    value = (text or "").strip()
    if not value:
        return "", notes

    # Remove invisible/control chars that may destabilize phonemization.
    value = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", value)

    # Remove non-Latin scripts to keep Vietnamese/Latin input stable.
    non_latin_pattern = r"[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]"
    if re.search(non_latin_pattern, value):
        value = re.sub(non_latin_pattern, " ", value)
        notes.append("Da loai bo ky tu he chu khong phai Latin de tranh doc sai.")

    # Keep common punctuation and Vietnamese Latin letters.
    value = re.sub(r"[^0-9A-Za-zÀ-ỹĐđ\s\.,!?;:'\"()\-\n/%&+=]", " ", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return value, notes


def apply_generation_guardrails(
    text: str,
    temperature: float,
    max_chars: int,
    top_k: int,
) -> tuple[str, float, int, int, list[str]]:
    clean_text, notes = sanitize_input_text(text)
    if not clean_text:
        return "", float(temperature), int(max_chars), int(top_k), notes

    safe_temperature = float(temperature)
    safe_max_chars = int(max_chars)
    safe_top_k = int(top_k)

    # Long text is more likely to drift; use safer generation params.
    if len(clean_text) > 260:
        if safe_temperature > 0.8:
            safe_temperature = 0.8
            notes.append("Van ban dai: da giam temperature xuong 0.8 de han che doc lac noi dung.")
        if safe_max_chars > 256:
            safe_max_chars = 256
            notes.append("Van ban dai: da giam max_chars xuong 256 de tach doan ngan hon.")
        if safe_top_k > 35:
            safe_top_k = 35
            notes.append("Van ban dai: da giam top_k xuong 35 de on dinh hon.")

    safe_max_chars = max(128, min(1024, safe_max_chars))
    safe_top_k = max(1, min(100, safe_top_k))
    safe_temperature = max(0.1, min(1.5, safe_temperature))
    return clean_text, safe_temperature, safe_max_chars, safe_top_k, notes


def read_text_file_content(text_file: str) -> str:
    path = Path(text_file)
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="cp1258")


def _safe_repo_name(repo_or_path: str) -> str:
    return repo_or_path.strip().replace("/", "_").replace("\\", "_").replace(":", "_")


def _is_valid_merged_cache(cache_dir: Path) -> bool:
    required_files = ["config.json", "tokenizer_config.json", "voices.json"]
    return cache_dir.exists() and all((cache_dir / f).exists() for f in required_files)


def ensure_merged_lora_model(
    base_model: str,
    lora_model: str,
    codec_model: str,
    hf_token: str | None,
) -> str:
    MERGED_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_dir = MERGED_CACHE_ROOT / _safe_repo_name(lora_model)

    if _is_valid_merged_cache(cache_dir):
        return str(cache_dir.resolve())

    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    merge_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Merging LoRA once to cache: {cache_dir}")
    print(f"Merge device: {merge_device}")

    temp_tts = VieNeuTTS(
        backbone_repo=base_model,
        backbone_device=merge_device,
        codec_repo=codec_model,
        codec_device="cpu",
        hf_token=hf_token,
    )
    try:
        temp_tts.load_lora_adapter(lora_model, hf_token=hf_token)

        if not hasattr(temp_tts.backbone, "merge_and_unload"):
            raise RuntimeError("Backbone does not support merge_and_unload for LoRA.")

        temp_tts.backbone = temp_tts.backbone.merge_and_unload()
        temp_tts.backbone.save_pretrained(str(cache_dir))
        temp_tts.tokenizer.save_pretrained(str(cache_dir))

        try:
            from transformers import AutoTokenizer

            slow_tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, token=hf_token)
            slow_tokenizer.save_pretrained(str(cache_dir))
        except Exception as exc:
            print(f"Warning: could not save slow tokenizer files: {exc}")

        voices_content = {
            "meta": {"note": "Auto-generated from LoRA merge for FastVieNeuTTS"},
            "default_voice": temp_tts._default_voice,
            "presets": temp_tts._preset_voices,
        }
        (cache_dir / "voices.json").write_text(
            json.dumps(voices_content, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise
    finally:
        temp_tts.close()

    return str(cache_dir.resolve())


def release_model_instance(model: Any) -> None:
    if model is None:
        return
    if hasattr(model, "close") and callable(model.close):
        model.close()
    elif hasattr(model, "cleanup_memory") and callable(model.cleanup_memory):
        model.cleanup_memory()


def get_or_load_tts(
    base_model: str,
    lora_model: str,
    codec_model: str,
    device: str,
    hf_token: str | None,
    prefer_fast: bool = True,
    fast_max_batch_size: int = 8,
) -> tuple[Any, str, str | None, str | None]:
    global _MODEL_INSTANCE, _MODEL_SIGNATURE, _MODEL_BACKEND, _MODEL_MERGED_PATH, _MODEL_LAST_ERROR
    signature = (
        base_model,
        lora_model,
        codec_model,
        device,
        hf_token or "",
        prefer_fast,
        fast_max_batch_size,
    )

    with _MODEL_LOCK:
        if _MODEL_INSTANCE is not None and _MODEL_SIGNATURE == signature:
            return _MODEL_INSTANCE, _MODEL_BACKEND or "unknown", _MODEL_MERGED_PATH, _MODEL_LAST_ERROR

        if _MODEL_INSTANCE is not None:
            release_model_instance(_MODEL_INSTANCE)
            _MODEL_INSTANCE = None
            _MODEL_SIGNATURE = None
            _MODEL_BACKEND = None
            _MODEL_MERGED_PATH = None
            _MODEL_LAST_ERROR = None

        last_error = None
        merged_model_path = None
        can_use_fast = prefer_fast and device.startswith("cuda")
        if can_use_fast:
            try:
                merged_model_path = ensure_merged_lora_model(
                    base_model=base_model,
                    lora_model=lora_model,
                    codec_model=codec_model,
                    hf_token=hf_token,
                )
                print(f"Loading merged model with FastVieNeuTTS: {merged_model_path}")
                tts = FastVieNeuTTS(
                    backbone_repo=merged_model_path,
                    backbone_device=device,
                    codec_repo=codec_model,
                    codec_device=device,
                    memory_util=0.3,
                    tp=1,
                    enable_prefix_caching=True,
                    enable_triton=True,
                    max_batch_size=fast_max_batch_size,
                    hf_token=hf_token,
                )
                _MODEL_INSTANCE = tts
                _MODEL_SIGNATURE = signature
                _MODEL_BACKEND = "fast-lmdeploy"
                _MODEL_MERGED_PATH = merged_model_path
                _MODEL_LAST_ERROR = None
                return _MODEL_INSTANCE, _MODEL_BACKEND, _MODEL_MERGED_PATH, _MODEL_LAST_ERROR
            except Exception as exc:
                last_error = f"Fast backend failed, fallback to standard: {exc}"
                print(last_error)

        print(f"Loading base model (standard): {base_model} on {device}")
        print(f"Loading codec model (standard): {codec_model} on {device}")
        tts = VieNeuTTS(
            backbone_repo=base_model,
            backbone_device=device,
            codec_repo=codec_model,
            codec_device=device,
            hf_token=hf_token,
        )
        print(f"Loading LoRA adapter (standard): {lora_model}")
        tts.load_lora_adapter(lora_model, hf_token=hf_token)
        _MODEL_INSTANCE = tts
        _MODEL_SIGNATURE = signature
        _MODEL_BACKEND = "standard"
        _MODEL_MERGED_PATH = None
        _MODEL_LAST_ERROR = last_error
        return _MODEL_INSTANCE, _MODEL_BACKEND, _MODEL_MERGED_PATH, _MODEL_LAST_ERROR


def close_model() -> None:
    global _MODEL_INSTANCE, _MODEL_SIGNATURE, _MODEL_BACKEND, _MODEL_MERGED_PATH, _MODEL_LAST_ERROR
    with _MODEL_LOCK:
        if _MODEL_INSTANCE is not None:
            release_model_instance(_MODEL_INSTANCE)
        _MODEL_INSTANCE = None
        _MODEL_SIGNATURE = None
        _MODEL_BACKEND = None
        _MODEL_MERGED_PATH = None
        _MODEL_LAST_ERROR = None


def collect_runtime_info(requested_device: str) -> str:
    resolved = detect_device(requested_device)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    fast_ready = resolved.startswith("cuda")
    active_backend = _MODEL_BACKEND or "not_loaded"
    return (
        f"torch={torch.__version__} | cuda_available={torch.cuda.is_available()} | "
        f"gpu={gpu_name} | requested_device={requested_device} | resolved_device={resolved} | "
        f"fast_backend_ready={fast_ready} | active_backend={active_backend}"
    )


def get_model_runtime_devices(tts: Any) -> tuple[str, str]:
    backbone_device = "unknown"
    codec_device = "unknown"

    try:
        if hasattr(tts, "gen_config"):
            backbone_device = "cuda (LMDeploy)"
        elif hasattr(tts.backbone, "device"):
            backbone_device = str(tts.backbone.device)
        elif hasattr(tts.backbone, "parameters"):
            backbone_device = str(next(tts.backbone.parameters()).device)
    except Exception:
        backbone_device = "unknown"

    try:
        if hasattr(tts.codec, "device"):
            codec_device = str(tts.codec.device)
        elif hasattr(tts.codec, "parameters"):
            codec_device = str(next(tts.codec.parameters()).device)
    except Exception:
        codec_device = "unknown"

    return backbone_device, codec_device


def save_audio_output(tts: Any, audio: Any, output_path: str | Path) -> None:
    target_path = str(output_path)
    if hasattr(tts, "save") and callable(getattr(tts, "save")):
        tts.save(audio, target_path)
        return
    sample_rate = int(getattr(tts, "sample_rate", 24_000))
    sf.write(target_path, audio, sample_rate)


def apply_audio_post_processing(
    audio: Any,
    sample_rate: int,
    pitch_steps: float,
    speed_factor: float,
) -> np.ndarray:
    processed = np.asarray(audio, dtype=np.float32).squeeze()
    if processed.ndim != 1:
        raise ValueError("Audio dau ra khong dung dinh dang 1 kenh de xu ly hau ky.")

    needs_pitch = abs(float(pitch_steps)) > 1e-3
    needs_speed = abs(float(speed_factor) - 1.0) > 1e-3
    if not needs_pitch and not needs_speed:
        return processed

    if librosa is None:
        raise RuntimeError("Chua co librosa de chinh cao do va toc do doc.")

    if needs_pitch:
        processed = librosa.effects.pitch_shift(
            processed,
            sr=int(sample_rate),
            n_steps=float(pitch_steps),
        )
    if needs_speed:
        processed = librosa.effects.time_stretch(
            processed,
            rate=max(0.5, min(1.8, float(speed_factor))),
        )

    peak = float(np.max(np.abs(processed))) if processed.size else 0.0
    if peak > 1.0:
        processed = processed / peak
    return processed.astype(np.float32, copy=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Text-to-Voice voi model LoRA Ngoc Huyen (VieNeu-TTS-0.3B)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ui", "cli"],
        default="ui",
        help="Che do chay. Mac dinh la ui de bam Run trong VS Code.",
    )
    parser.add_argument("--text", type=str, default=None, help="Noi dung can chuyen thanh giong noi.")
    parser.add_argument("--text-file", type=str, default=None, help="Duong dan file .txt dau vao.")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ngoc_huyen_tts.wav",
        help="Duong dan file wav dau ra (chi dung cho mode cli).",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="ID preset voice trong voices.json cua model LoRA.",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="In danh sach preset voices sau khi nap model (chi dung cho mode cli).",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="File audio mau de clone giong (tuy chon).",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Noi dung dung voi audio mau, bat buoc khi dung --ref-audio.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Model base (khuyen nghi: pnnbao-ump/VieNeu-TTS-0.3B).",
    )
    parser.add_argument(
        "--lora-model",
        type=str,
        default=DEFAULT_LORA_MODEL,
        help="Repo LoRA model.",
    )
    parser.add_argument(
        "--codec-model",
        type=str,
        default=DEFAULT_CODEC_MODEL,
        help="Repo codec model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Thiet bi suy luan.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Nhiet do sampling.")
    parser.add_argument("--max-chars", type=int, default=384, help="So ky tu toi da moi chunk.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="Dieu chinh cao do hau ky theo semitone. 0.0 la mac dinh.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Dieu chinh toc do doc hau ky. 1.0 la mac dinh, >1.0 nhanh hon.",
    )
    parser.add_argument(
        "--fast-max-batch-size",
        type=int,
        default=8,
        help="Batch size cho FastVieNeuTTS (LMDeploy).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (neu model private).",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Dia chi app cho mode ui.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7861,
        help="Cong app cho mode ui.",
    )
    return parser


def run_cli(args: argparse.Namespace) -> None:
    if args.ref_audio and not args.ref_text:
        raise ValueError("Can --ref-text khi su dung --ref-audio.")

    text_input = read_text_input(args.text, args.text_file)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = detect_device(args.device)
    tts, backend, merged_path, last_error = get_or_load_tts(
        base_model=args.base_model,
        lora_model=args.lora_model,
        codec_model=args.codec_model,
        device=device,
        hf_token=args.hf_token,
        prefer_fast=True,
        fast_max_batch_size=args.fast_max_batch_size,
    )
    print(f"Backend: {backend}")
    if merged_path:
        print(f"Merged model cache: {merged_path}")
    if last_error:
        print(last_error)

    voices = tts.list_preset_voices()
    if args.list_voices:
        print("\nPreset voices:")
        if not voices:
            print("  (Khong co voice preset trong model)")
        else:
            for desc, voice_id in voices:
                print(f"  - {voice_id}: {desc}")

    text_for_infer, safe_temp, safe_max_chars, safe_top_k, notes = apply_generation_guardrails(
        text=text_input,
        temperature=args.temperature,
        max_chars=args.max_chars,
        top_k=args.top_k,
    )
    if not text_for_infer:
        raise ValueError("Van ban dau vao rong hoac khong hop le sau khi tien xu ly.")
    for note in notes:
        print(f"Note: {note}")

    if args.ref_audio:
        print(f"Synthesizing with reference audio: {args.ref_audio}")
        audio = tts.infer(
            text=text_for_infer,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            temperature=safe_temp,
            max_chars=safe_max_chars,
            top_k=safe_top_k,
            silence_p=0.08,
        )
    elif args.voice:
        print(f"Synthesizing with preset voice: {args.voice}")
        voice_data = tts.get_preset_voice(args.voice)
        audio = tts.infer(
            text=text_for_infer,
            voice=voice_data,
            temperature=safe_temp,
            max_chars=safe_max_chars,
            top_k=safe_top_k,
            silence_p=0.08,
        )
    else:
        print("Synthesizing with default voice from loaded model...")
        audio = tts.infer(
            text=text_for_infer,
            temperature=safe_temp,
            max_chars=safe_max_chars,
            top_k=safe_top_k,
            silence_p=0.08,
        )

    audio = apply_audio_post_processing(
        audio=audio,
        sample_rate=int(getattr(tts, "sample_rate", 24_000)),
        pitch_steps=args.pitch,
        speed_factor=args.speed,
    )
    save_audio_output(tts, audio, output_path)
    print(f"Saved audio: {output_path}")


def run_ui(args: argparse.Namespace) -> None:
    initial_device = args.device if args.device in get_supported_devices() else "auto"
    initial_text = (
        "Xin chao, day la ung dung text to speech bang model "
        "pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen."
    )

    def load_text_file(file_path: str | None):
        if not file_path:
            return gr.update(), "Chua chon file van ban."
        try:
            text = read_text_file_content(file_path)
            return text, f"Da nap file: {file_path}"
        except Exception as exc:
            return gr.update(), f"Loi doc file: {exc}"

    def synthesize_text(
        text: str,
        voice_id: str,
        temperature: float,
        max_chars: int,
        top_k: int,
        pitch_steps: float,
        speed_factor: float,
        device_choice: str,
        fast_batch_size: int,
    ):
        if not text or not text.strip():
            return None, "Vui long nhap van ban hoac chon file van ban."
        try:
            resolved_device = detect_device(device_choice)
            tts, backend, merged_path, last_error = get_or_load_tts(
                base_model=args.base_model,
                lora_model=args.lora_model,
                codec_model=args.codec_model,
                device=resolved_device,
                hf_token=args.hf_token,
                prefer_fast=True,
                fast_max_batch_size=int(fast_batch_size),
            )

            text_for_infer, safe_temp, safe_max_chars, safe_top_k, notes = apply_generation_guardrails(
                text=text.strip(),
                temperature=temperature,
                max_chars=int(max_chars),
                top_k=int(top_k),
            )
            if not text_for_infer:
                return None, "Van ban dau vao rong hoac khong hop le sau khi tien xu ly."

            if voice_id and voice_id.strip():
                voice_data = tts.get_preset_voice(voice_id.strip())
                audio = tts.infer(
                    text=text_for_infer,
                    voice=voice_data,
                    temperature=safe_temp,
                    max_chars=safe_max_chars,
                    top_k=safe_top_k,
                    silence_p=0.08,
                )
            else:
                audio = tts.infer(
                    text=text_for_infer,
                    temperature=safe_temp,
                    max_chars=safe_max_chars,
                    top_k=safe_top_k,
                    silence_p=0.08,
                )

            audio = apply_audio_post_processing(
                audio=audio,
                sample_rate=int(getattr(tts, "sample_rate", 24_000)),
                pitch_steps=float(pitch_steps),
                speed_factor=float(speed_factor),
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                output_path = temp_file.name

            save_audio_output(tts, audio, output_path)
            backbone_device, codec_device = get_model_runtime_devices(tts)
            status = (
                f"Tao audio thanh cong: {output_path}\n"
                f"Requested={device_choice} | Resolved={resolved_device} | "
                f"Backend={backend} | Backbone={backbone_device} | Codec={codec_device}\n"
                f"CaoDo={float(pitch_steps):+.1f} st | TocDo={float(speed_factor):.2f}x"
            )
            if merged_path:
                status += f"\nMergedCache={merged_path}"
            if last_error:
                status += f"\nNote={last_error}"
            if notes:
                status += "\n" + "\n".join(f"Note={item}" for item in notes)
            return output_path, status
        except Exception as exc:
            return None, f"Loi tao audio: {exc}"

    with gr.Blocks(title="Ngoc Huyen Text-to-Voice") as app:
        gr.Markdown("## Ngoc Huyen Text-to-Voice")
        gr.Markdown(
            "Bam Run trong VS Code de mo app nay. "
            "Ban co the nhap van ban truc tiep hoac chon file `.txt`."
        )
        gr.Markdown(
            "Toi uu toc do: app uu tien FastVieNeuTTS (LMDeploy) tren CUDA va "
            "tu dong merge LoRA vao cache 1 lan."
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Van ban dau vao",
                    lines=8,
                    value=initial_text,
                )
                file_input = gr.File(
                    label="Chon file van ban (.txt)",
                    file_types=[".txt"],
                    type="filepath",
                )
                voice_input = gr.Textbox(
                    label="Voice ID (tuy chon)",
                    placeholder="De trong de dung default voice cua model",
                    info="Nhap ID voice preset neu muon doi chat giong.",
                )
                device_input = gr.Radio(
                    choices=get_supported_devices(),
                    value=initial_device,
                    label="Thiet bi su dung",
                    info="Chon cuda de ep dung GPU NVIDIA.",
                )
                temperature_input = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=args.temperature,
                    step=0.1,
                    label="Do ngau nhien (temperature)",
                    info="Tang len neu muon giong doc bien thien hon; giam xuong de on dinh hon.",
                )
                max_chars_input = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=args.max_chars,
                    step=64,
                    label="Do dai moi doan (max chars)",
                    info="Van ban dai se duoc tach thanh nhieu doan nho de tranh loi.",
                )
                top_k_input = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=args.top_k,
                    step=1,
                    label="Do rong lua chon (top-k)",
                    info="Gia tri nho hon thuong an toan hon, gia tri lon tu nhien hon.",
                )
                pitch_input = gr.Slider(
                    minimum=-12.0,
                    maximum=12.0,
                    value=args.pitch,
                    step=0.5,
                    label="Cao do giong (semitone)",
                    info="So duong lam giong cao hon, so am lam giong tram hon.",
                )
                speed_input = gr.Slider(
                    minimum=0.7,
                    maximum=1.5,
                    value=args.speed,
                    step=0.05,
                    label="Toc do doc",
                    info="1.0 la mac dinh, >1.0 nhanh hon, <1.0 cham hon.",
                )
                fast_batch_input = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=args.fast_max_batch_size,
                    step=1,
                    label="Fast LMDeploy batch size",
                    info="Chi anh huong khi backend fast-lmdeploy duoc su dung.",
                )
            with gr.Column(scale=2):
                gr.Markdown(
                    "### Bang giai thich chuc nang\n"
                    "- **Van ban dau vao**: Noi dung se duoc doc thanh giong noi.\n"
                    "- **Chon file van ban**: Nap nhanh file `.txt` vao o nhap.\n"
                    "- **Voice ID**: Doi sang voice preset cua model, de trong se dung voice mac dinh.\n"
                    "- **Thiet bi su dung**: `cuda` nhanh nhat neu may co GPU NVIDIA.\n"
                    "- **Temperature / Top-k**: Chinh muc do bien thien khi sinh giong.\n"
                    "- **Max chars**: Chia nho van ban dai de giam loi doc lac.\n"
                    "- **Cao do giong**: Tang/giam cao do sau khi da tao audio.\n"
                    "- **Toc do doc**: Lam audio nhanh hon hoac cham hon ma van giu chat giong tuong doi."
                )

        with gr.Row():
            btn_load = gr.Button("Nap file vao o van ban")
            btn_generate = gr.Button("Tao giong noi", variant="primary")
            btn_check_runtime = gr.Button("Kiem tra runtime")

        audio_output = gr.Audio(label="Audio ket qua", type="filepath")
        runtime_info_output = gr.Textbox(label="Thong tin runtime", lines=3, interactive=False)
        status_output = gr.Textbox(label="Trang thai", lines=6)

        btn_load.click(
            fn=load_text_file,
            inputs=[file_input],
            outputs=[text_input, status_output],
        )
        btn_generate.click(
            fn=synthesize_text,
            inputs=[
                text_input,
                voice_input,
                temperature_input,
                max_chars_input,
                top_k_input,
                pitch_input,
                speed_input,
                device_input,
                fast_batch_input,
            ],
            outputs=[audio_output, status_output],
        )
        btn_check_runtime.click(
            fn=collect_runtime_info,
            inputs=[device_input],
            outputs=[runtime_info_output],
        )
        app.load(
            fn=collect_runtime_info,
            inputs=[device_input],
            outputs=[runtime_info_output],
        )

    app.queue().launch(server_name=args.server_name, server_port=args.server_port, share=False)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli(args)
    else:
        run_ui(args)


atexit.register(close_model)


if __name__ == "__main__":
    main()
