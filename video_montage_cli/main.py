#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
SUPPORTED_OUTPUT_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
SPEECH_MIN_SHOT_SECONDS = 2.0
SPEECH_MAX_SHOT_SECONDS = 6.5
WATERMARK_POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
except Exception:
    tk = None
    filedialog = None
    messagebox = None
    simpledialog = None
    ttk = None


def parse_size(value: str) -> tuple[int, int]:
    value = value.strip().lower()
    if "x" not in value:
        raise argparse.ArgumentTypeError("Size must be in WxH format, example: 1920x1080")

    parts = value.split("x", 1)
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise argparse.ArgumentTypeError("Size must be in WxH format, example: 1080x1920")

    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive integers")

    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a final video from an audio file and a folder of video shots."
    )
    parser.add_argument("--gui", action="store_true", help="Launch GUI for batch processing")
    parser.add_argument("--audio", type=Path, help="Input audio file path")
    parser.add_argument("--videos", type=Path, help="Input videos folder")
    parser.add_argument("--out", type=Path, help="Output final MP4 path")
    parser.add_argument("--mode", default="even", choices=("even", "speech"), help="Cut mode")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Video encoding device. auto prefers GPU (NVENC) when available.",
    )
    parser.add_argument(
        "--shot-len",
        default=4.0,
        type=float,
        help="Shot length in seconds, only used for even mode (default: 4.0)",
    )
    parser.add_argument(
        "--size",
        default=parse_size("1920x1080"),
        type=parse_size,
        help="Output size (example: 1920x1080 or 1080x1920)",
    )
    parser.add_argument("--fps", default=30, type=int, help="Output FPS (default: 30)")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle input video order before building shots",
    )
    parser.add_argument("--watermark", type=Path, help="Watermark PNG path")
    parser.add_argument(
        "--wm-scale",
        default=0.12,
        type=float,
        help="Watermark width ratio relative to video width (default: 0.12)",
    )
    parser.add_argument(
        "--wm-pos",
        default="bottom-right",
        choices=WATERMARK_POSITIONS,
        help="Watermark position (default: bottom-right)",
    )
    parser.add_argument("--bgm", type=Path, help="Background music file path")
    parser.add_argument(
        "--bgm-volume",
        default=0.12,
        type=float,
        help="BGM base volume when voice is not speaking (default: 0.12)",
    )
    parser.add_argument(
        "--bgm-duck-volume",
        default=0.035,
        type=float,
        help="BGM ducked volume when voice is speaking (default: 0.035)",
    )
    return parser


def list_video_files(videos_dir: Path) -> list[Path]:
    files = [p for p in sorted(videos_dir.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS]
    return files


def list_audio_files(audio_dir: Path) -> list[Path]:
    files = [p for p in sorted(audio_dir.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS]
    return files


def list_output_video_files(output_dir: Path) -> list[Path]:
    files = [p for p in sorted(output_dir.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_OUTPUT_VIDEO_EXTS]
    return files


def open_with_default_app(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
        return
    subprocess.Popen(["xdg-open", str(path)])


def require_binaries() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "Missing required binaries: "
            + ", ".join(missing)
            + ". Install FFmpeg and ensure ffmpeg/ffprobe are in PATH."
        )


def is_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def ffmpeg_has_encoder(encoder_name: str) -> bool:
    cmd = ["ffmpeg", "-hide_banner", "-encoders"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return False
    text = f"{proc.stdout}\n{proc.stderr}"
    return encoder_name in text


def resolve_video_encoder(device_choice: str) -> tuple[str, str]:
    device_choice = (device_choice or "auto").strip().lower()
    if device_choice not in {"auto", "cuda", "cpu"}:
        raise RuntimeError("--device must be one of: auto, cuda, cpu")

    cuda_ok = is_cuda_available()
    nvenc_ok = ffmpeg_has_encoder("h264_nvenc")

    if device_choice == "cpu":
        return "libx264", "cpu"

    if device_choice == "cuda":
        if not cuda_ok:
            raise RuntimeError("CUDA is not available in current Python runtime.")
        if not nvenc_ok:
            raise RuntimeError("FFmpeg does not support h264_nvenc. Install FFmpeg build with NVENC.")
        return "h264_nvenc", "cuda"

    if cuda_ok and nvenc_ok:
        return "h264_nvenc", "cuda"
    return "libx264", "cpu"


def get_media_duration_seconds(media_path: Path, media_label: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {media_label} file: {media_path}")

    try:
        duration = float(proc.stdout.strip())
    except ValueError as err:
        raise RuntimeError(f"Cannot read {media_label} duration from ffprobe output") from err

    if duration <= 0:
        raise RuntimeError(f"{media_label.capitalize()} duration must be > 0 seconds")
    return duration


def get_audio_duration_seconds(audio_path: Path) -> float:
    return get_media_duration_seconds(audio_path, "audio")


def get_video_duration_seconds(video_path: Path) -> float:
    return get_media_duration_seconds(video_path, "video")


def to_ffconcat_path(path: Path) -> str:
    # Use absolute POSIX-like path to avoid backslash escaping issues on Windows.
    return path.resolve().as_posix().replace("'", "'\\''")


def run_cmd(cmd: list[str], err_msg: str) -> None:
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(err_msg)


def unique_destination_path(dest_dir: Path, src_name: str) -> Path:
    candidate = dest_dir / src_name
    if not candidate.exists():
        return candidate

    stem = Path(src_name).stem
    suffix = Path(src_name).suffix
    idx = 1
    while True:
        candidate = dest_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def import_files_to_dir(src_paths: list[str], dest_dir: Path, allowed_exts: set[str]) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    imported: list[Path] = []
    for src in src_paths:
        src_path = Path(src)
        if not src_path.is_file():
            continue
        if src_path.suffix.lower() not in allowed_exts:
            continue
        out_path = unique_destination_path(dest_dir, src_path.name)
        shutil.copy2(src_path, out_path)
        imported.append(out_path)
    return imported


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.audio is None:
        raise RuntimeError("--audio is required")
    if args.videos is None:
        raise RuntimeError("--videos is required")
    if args.out is None:
        raise RuntimeError("--out is required")
    if not args.audio.is_file():
        raise RuntimeError(f"--audio not found: {args.audio}")
    if not args.videos.is_dir():
        raise RuntimeError(f"--videos is not a folder: {args.videos}")
    if args.mode not in {"even", "speech"}:
        raise RuntimeError("--mode must be 'even' or 'speech'")
    device_choice = getattr(args, "device", "auto")
    if device_choice not in {"auto", "cuda", "cpu"}:
        raise RuntimeError("--device must be one of: auto, cuda, cpu")
    if len(args.size) != 2 or args.size[0] <= 0 or args.size[1] <= 0:
        raise RuntimeError("--size must be two positive integers")
    if args.fps <= 0:
        raise RuntimeError("--fps must be > 0")
    if args.shot_len <= 0:
        raise RuntimeError("--shot-len must be > 0")
    if args.wm_scale <= 0 or args.wm_scale > 1:
        raise RuntimeError("--wm-scale must be > 0 and <= 1")
    if args.wm_pos not in WATERMARK_POSITIONS:
        raise RuntimeError(f"--wm-pos must be one of: {', '.join(WATERMARK_POSITIONS)}")
    if args.bgm_volume < 0 or args.bgm_volume > 1:
        raise RuntimeError("--bgm-volume must be in [0, 1]")
    if args.bgm_duck_volume < 0 or args.bgm_duck_volume > 1:
        raise RuntimeError("--bgm-duck-volume must be in [0, 1]")
    if args.watermark is not None:
        if not args.watermark.is_file():
            raise RuntimeError(f"--watermark not found: {args.watermark}")
        if args.watermark.suffix.lower() != ".png":
            raise RuntimeError("--watermark must be a .png file")
    if args.bgm is not None and not args.bgm.is_file():
        raise RuntimeError(f"--bgm not found: {args.bgm}")

    video_files = list_video_files(args.videos)
    if not video_files:
        raise RuntimeError(
            f"No video files found in {args.videos} (supported: {', '.join(sorted(SUPPORTED_VIDEO_EXTS))})"
        )


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    try:
        validate_runtime_args(args)
    except RuntimeError as err:
        parser.error(str(err))


def build_even_order(video_files: list[Path], shots_needed: int, shuffle: bool) -> list[Path]:
    if not video_files:
        return []

    base = video_files[:]
    if shuffle:
        random.shuffle(base)

    ordered: list[Path] = []
    idx = 0
    while len(ordered) < shots_needed:
        ordered.append(base[idx % len(base)])
        idx += 1
    return ordered


def build_full_video_sequence(
    video_files: list[Path],
    target_duration: float,
    shuffle: bool,
) -> tuple[list[Path], list[float]]:
    if not video_files:
        return [], []
    if target_duration <= 0:
        return [], []

    play_order = video_files[:]
    if shuffle:
        random.shuffle(play_order)

    duration_cache: dict[Path, float] = {}
    shot_sources: list[Path] = []
    shot_lengths: list[float] = []
    remaining = target_duration
    idx = 0

    while remaining > 0.001:
        src = play_order[idx % len(play_order)]
        if src not in duration_cache:
            duration_cache[src] = get_video_duration_seconds(src)
        src_duration = duration_cache[src]
        use_len = min(src_duration, remaining)
        shot_sources.append(src)
        shot_lengths.append(use_len)
        remaining -= use_len
        idx += 1

    return shot_sources, shot_lengths


def transcode_shot(
    src_video: Path,
    out_shot: Path,
    shot_len: float,
    width: int,
    height: int,
    fps: int,
    video_encoder: str = "libx264",
) -> None:
    # Fill frame then center-crop to exact target size.
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},fps={fps},setsar=1"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(src_video),
        "-t",
        f"{shot_len:.3f}",
        "-an",
        "-vf",
        vf,
        "-c:v",
        video_encoder,
        "-pix_fmt",
        "yuv420p",
    ]
    if video_encoder == "h264_nvenc":
        cmd.extend(["-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0"])
    cmd.append(str(out_shot))
    run_cmd(cmd, f"ffmpeg failed while creating shot from: {src_video}")


def build_concat_list(shot_files: list[Path], concat_path: Path) -> None:
    with concat_path.open("w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for shot in shot_files:
            f.write(f"file '{to_ffconcat_path(shot)}'\n")


def mux_with_audio(
    concat_path: Path,
    audio_path: Path,
    out_path: Path,
    audio_seconds: float,
    watermark_path: Path | None = None,
    watermark_scale: float = 0.12,
    watermark_pos: str = "bottom-right",
    bgm_path: Path | None = None,
    bgm_volume: float = 0.12,
    bgm_duck_volume: float = 0.035,
    speech_segments: list[tuple[float, float]] | None = None,
    video_encoder: str = "libx264",
) -> None:
    wm_input_idx: int | None = None
    bgm_input_idx: int | None = None
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-i",
        str(audio_path),
    ]

    if bgm_path is not None:
        bgm_input_idx = 2
        cmd.extend(["-stream_loop", "-1", "-i", str(bgm_path)])

    if watermark_path is not None:
        wm_input_idx = 3 if bgm_input_idx is not None else 2
        cmd.extend(["-i", str(watermark_path)])

    filter_parts: list[str] = []
    video_map = "0:v:0"
    audio_map = "1:a:0"

    if wm_input_idx is not None:
        overlay_xy = {
            "top-left": "20:20",
            "top-right": "W-w-20:20",
            "bottom-left": "20:H-h-20",
            "bottom-right": "W-w-20:H-h-20",
        }[watermark_pos]
        filter_parts.append(
            f"[{wm_input_idx}:v][0:v]scale2ref=w=main_w*{watermark_scale:.4f}:h=-1[wm][base]"
        )
        filter_parts.append(f"[base][wm]overlay={overlay_xy}[vout]")
        video_map = "[vout]"

    if bgm_input_idx is not None:
        if speech_segments:
            cond = "+".join([f"between(t,{start:.3f},{end:.3f})" for start, end in speech_segments])
            volume_expr = f"if(gte({cond},1),{bgm_duck_volume:.4f},{bgm_volume:.4f})"
            filter_parts.append(
                f"[{bgm_input_idx}:a]volume='{volume_expr}',atrim=0:{audio_seconds:.3f}[bgmduck]"
            )
        else:
            filter_parts.append(f"[{bgm_input_idx}:a]volume={bgm_volume:.4f},atrim=0:{audio_seconds:.3f}[bgmduck]")
        filter_parts.append("[1:a][bgmduck]amix=inputs=2:duration=first:dropout_transition=0[aout]")
        audio_map = "[aout]"

    if filter_parts:
        cmd.extend(["-filter_complex", ";".join(filter_parts)])

    cmd.extend(
        [
            "-map",
            video_map,
            "-map",
            audio_map,
            "-c:v",
            video_encoder,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-t",
            f"{audio_seconds:.3f}",
        ]
    )
    if video_encoder == "h264_nvenc":
        cmd.extend(["-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0"])
    cmd.append(str(out_path))
    run_cmd(cmd, "ffmpeg failed while muxing final video and audio")


def render_fixed_shot_mode(
    args: argparse.Namespace,
    shot_sources: list[Path],
    shot_lengths: list[float],
    audio_seconds: float,
    speech_segments_for_ducking: list[tuple[float, float]] | None = None,
) -> None:
    if not shot_sources or not shot_lengths:
        raise RuntimeError("No source video available for rendering")
    if len(shot_sources) != len(shot_lengths):
        raise RuntimeError("Shot source count and shot length count do not match")
    video_encoder = getattr(args, "video_encoder", "libx264")

    with tempfile.TemporaryDirectory(prefix="video_montage_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        shot_files: list[Path] = []
        for idx, (src, shot_len) in enumerate(zip(shot_sources, shot_lengths)):
            shot_out = temp_dir / f"shot_{idx:05d}.mp4"
            transcode_shot(
                src_video=src,
                out_shot=shot_out,
                shot_len=shot_len,
                width=args.size[0],
                height=args.size[1],
                fps=args.fps,
                video_encoder=video_encoder,
            )
            shot_files.append(shot_out)

        concat_path = temp_dir / "shots.ffconcat"
        build_concat_list(shot_files, concat_path)
        mux_with_audio(
            concat_path=concat_path,
            audio_path=args.audio,
            out_path=args.out,
            audio_seconds=audio_seconds,
            watermark_path=args.watermark,
            watermark_scale=args.wm_scale,
            watermark_pos=args.wm_pos,
            bgm_path=args.bgm,
            bgm_volume=args.bgm_volume,
            bgm_duck_volume=args.bgm_duck_volume,
            speech_segments=speech_segments_for_ducking,
            video_encoder=video_encoder,
        )


def render_even_mode(
    args: argparse.Namespace,
    video_files: list[Path],
    audio_seconds: float,
    speech_segments_for_ducking: list[tuple[float, float]] | None = None,
) -> None:
    shot_sources, shot_lengths = build_full_video_sequence(
        video_files=video_files,
        target_duration=audio_seconds,
        shuffle=args.shuffle,
    )
    render_fixed_shot_mode(
        args=args,
        shot_sources=shot_sources,
        shot_lengths=shot_lengths,
        audio_seconds=audio_seconds,
        speech_segments_for_ducking=speech_segments_for_ducking,
    )


def split_segment_duration(duration: float, max_len: float) -> list[float]:
    pieces: list[float] = []
    remain = duration
    while remain > max_len:
        pieces.append(max_len)
        remain -= max_len
    if remain > 0:
        pieces.append(remain)
    return pieces


def normalize_segment_lengths(
    raw_durations: list[float],
    min_len: float = SPEECH_MIN_SHOT_SECONDS,
    max_len: float = SPEECH_MAX_SHOT_SECONDS,
) -> list[float]:
    if min_len <= 0 or max_len <= 0 or min_len > max_len:
        raise RuntimeError("Invalid speech normalization constraints")

    split_durations: list[float] = []
    for dur in raw_durations:
        if dur <= 0:
            continue
        split_durations.extend(split_segment_duration(dur, max_len))

    if not split_durations:
        return []

    normalized: list[float] = []
    i = 0
    while i < len(split_durations):
        current = split_durations[i]
        if current >= min_len:
            normalized.append(min(current, max_len))
            i += 1
            continue

        merged = current
        j = i + 1
        while merged < min_len and j < len(split_durations):
            merged += split_durations[j]
            j += 1

        if merged <= max_len:
            normalized.append(merged)
            i = j
            continue

        if normalized and (normalized[-1] + current) <= max_len:
            normalized[-1] += current
            i += 1
            continue

        normalized.append(min_len)
        i += 1

    for idx in range(len(normalized)):
        if normalized[idx] < min_len:
            normalized[idx] = min_len
        if normalized[idx] > max_len:
            normalized[idx] = max_len

    return normalized


def get_whisper_speech_segments(audio_path: Path, audio_seconds: float) -> list[tuple[float, float]]:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError("faster-whisper is not installed") from exc

    preferred_device = "cpu"
    preferred_compute_type = "int8"
    try:
        import torch

        if torch.cuda.is_available():
            preferred_device = "cuda"
            preferred_compute_type = "float16"
    except Exception:
        pass

    attempts: list[tuple[str, str]] = [(preferred_device, preferred_compute_type)]
    if preferred_device != "cpu":
        attempts.append(("cpu", "int8"))

    segments = None
    last_error: Exception | None = None
    for device, compute_type in attempts:
        try:
            model = WhisperModel("small", device=device, compute_type=compute_type)
            segments, _info = model.transcribe(str(audio_path), vad_filter=True)
            break
        except Exception as exc:
            last_error = exc
            if device != "cpu":
                print(f"[whisper] GPU failed, fallback to CPU: {exc}", file=sys.stderr)
            continue

    if segments is None:
        raise RuntimeError(f"Cannot initialize Whisper model: {last_error}")

    raw_segments: list[tuple[float, float]] = []
    for seg in segments:
        start = float(seg.start or 0.0)
        end = float(seg.end or 0.0)
        start = max(0.0, min(start, audio_seconds))
        end = max(start, min(end, audio_seconds))
        if (end - start) > 0:
            raw_segments.append((start, end))

    if not raw_segments:
        return []

    raw_segments.sort(key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    for start, end in raw_segments:
        if not merged:
            merged.append((start, end))
            continue
        last_start, last_end = merged[-1]
        if start <= (last_end + 0.05):
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def durations_from_segments(segments: list[tuple[float, float]]) -> list[float]:
    out: list[float] = []
    for start, end in segments:
        dur = end - start
        if dur > 0:
            out.append(dur)
    return out


def extend_shot_lengths_to_duration(
    shot_lengths: list[float],
    target_duration: float,
    min_len: float = SPEECH_MIN_SHOT_SECONDS,
    max_len: float = SPEECH_MAX_SHOT_SECONDS,
) -> list[float]:
    if not shot_lengths:
        return []

    total = sum(shot_lengths)
    while total < target_duration:
        remain = target_duration - total
        piece = remain
        if piece < min_len:
            piece = min_len
        if piece > max_len:
            piece = max_len
        shot_lengths.append(piece)
        total += piece
    return shot_lengths


def render_speech_mode(
    args: argparse.Namespace,
    video_files: list[Path],
    audio_seconds: float,
    speech_segments: list[tuple[float, float]],
) -> None:
    shot_lengths = normalize_segment_lengths(durations_from_segments(speech_segments))
    if not shot_lengths:
        raise RuntimeError("No speech segments extracted")

    shot_lengths = extend_shot_lengths_to_duration(shot_lengths, audio_seconds)
    shot_sources = build_even_order(video_files, len(shot_lengths), args.shuffle)
    render_fixed_shot_mode(
        args=args,
        shot_sources=shot_sources,
        shot_lengths=shot_lengths,
        audio_seconds=audio_seconds,
        speech_segments_for_ducking=speech_segments,
    )


def execute_render(args: argparse.Namespace) -> tuple[Path, list[str]]:
    if not hasattr(args, "device"):
        args.device = "auto"
    validate_runtime_args(args)
    require_binaries()

    video_files = list_video_files(args.videos)
    audio_seconds = get_audio_duration_seconds(args.audio)
    whisper_segments: list[tuple[float, float]] | None = None
    whisper_error: Exception | None = None
    logs: list[str] = []
    need_whisper = args.mode == "speech" or args.bgm is not None
    if need_whisper:
        try:
            whisper_segments = get_whisper_speech_segments(args.audio, audio_seconds)
            if not whisper_segments:
                raise RuntimeError("No speech segments extracted")
        except Exception as exc:
            whisper_error = exc
            whisper_segments = None

    selected_encoder, resolved_device = resolve_video_encoder(getattr(args, "device", "auto"))
    logs.append(
        f"[video] requested_device={getattr(args, 'device', 'auto')} | resolved_device={resolved_device} | encoder={selected_encoder}"
    )

    def _render_with_encoder(video_encoder: str) -> None:
        render_args = argparse.Namespace(**vars(args))
        render_args.video_encoder = video_encoder
        if render_args.mode == "even":
            if render_args.bgm is not None and whisper_error is not None:
                logs.append(f"[bgm] using fixed low volume (no ducking): {whisper_error}")
            render_even_mode(
                render_args, video_files, audio_seconds, speech_segments_for_ducking=whisper_segments
            )
        else:
            if whisper_segments is None:
                logs.append(f"[speech] fallback to even mode: {whisper_error}")
                render_even_mode(render_args, video_files, audio_seconds, speech_segments_for_ducking=None)
            else:
                render_speech_mode(render_args, video_files, audio_seconds, whisper_segments)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        _render_with_encoder(selected_encoder)
    except Exception as exc:
        if selected_encoder == "h264_nvenc":
            logs.append(f"[video] GPU encode failed, fallback CPU libx264: {exc}")
            _render_with_encoder("libx264")
        else:
            raise

    return args.out, logs


class VideoMontageGUI:
    def __init__(self, root: "tk.Tk", module_dir: Path) -> None:
        self.root = root
        self.module_dir = module_dir
        self.video_dir = module_dir / "input" / "videos"
        self.audio_dir = module_dir / "input" / "audio"
        self.output_dir = module_dir / "output"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mode_var = tk.StringVar(value="even")
        self.orientation_var = tk.StringVar(value="landscape")
        self.device_var = tk.StringVar(value="auto")
        self.fps_var = tk.StringVar(value="30")
        self.shot_len_var = tk.StringVar(value="4.0")
        self.shuffle_var = tk.BooleanVar(value=False)
        self.is_running = False

        self.video_files: list[Path] = []
        self.audio_files: list[Path] = []
        self.output_files: list[Path] = []

        self.root.title("Ghep Video Tu Dong")
        self.root.geometry("1380x660")
        self.root.minsize(1100, 600)
        self._build_ui()
        self._refresh_video_list()
        self._refresh_audio_list()
        self._refresh_output_list()

    def _build_ui(self) -> None:
        wrapper = ttk.Frame(self.root, padding=12)
        wrapper.pack(fill="both", expand=True)

        paths_frame = ttk.LabelFrame(wrapper, text="Thu muc lam viec")
        paths_frame.pack(fill="x", pady=(0, 10))
        self.video_path_label = ttk.Label(paths_frame, text=f"Thu muc video: {self.video_dir}")
        self.video_path_label.grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Button(paths_frame, text="Chon thu muc video", command=self._choose_video_dir).grid(
            row=0, column=1, padx=8, pady=6
        )
        self.audio_path_label = ttk.Label(paths_frame, text=f"Thu muc audio: {self.audio_dir}")
        self.audio_path_label.grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Button(paths_frame, text="Chon thu muc audio", command=self._choose_audio_dir).grid(
            row=1, column=1, padx=8, pady=6
        )
        self.output_path_label = ttk.Label(paths_frame, text=f"Thu muc xuat video: {self.output_dir}")
        self.output_path_label.grid(row=2, column=0, sticky="w", padx=8, pady=6)
        ttk.Button(paths_frame, text="Chon thu muc xuat", command=self._choose_output_dir).grid(
            row=2, column=1, padx=8, pady=6
        )

        content_frame = ttk.Panedwindow(wrapper, orient="horizontal")
        content_frame.pack(fill="both", expand=True)

        left_panel = ttk.Frame(content_frame)
        right_panel = ttk.Frame(content_frame)
        content_frame.add(left_panel, weight=7)
        content_frame.add(right_panel, weight=5)

        settings_frame = ttk.LabelFrame(left_panel, text="Bang tuy chinh")
        settings_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(settings_frame, text="Che do cat canh").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(settings_frame, textvariable=self.mode_var, values=("even", "speech"), width=12, state="readonly").grid(
            row=0, column=1, sticky="w", padx=8, pady=6
        )
        ttk.Label(settings_frame, text="even: chay het tung video | speech: cat theo giong noi").grid(
            row=0, column=2, columnspan=3, sticky="w", padx=8, pady=6
        )

        ttk.Label(settings_frame, text="Ty le man hinh").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Radiobutton(
            settings_frame, text="Ngang (1920x1080)", value="landscape", variable=self.orientation_var
        ).grid(row=1, column=1, sticky="w", padx=8, pady=6)
        ttk.Radiobutton(
            settings_frame, text="Doc (1080x1920)", value="portrait", variable=self.orientation_var
        ).grid(row=1, column=2, sticky="w", padx=8, pady=6)

        ttk.Label(settings_frame, text="FPS (khung hinh/giay)").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(settings_frame, textvariable=self.fps_var, width=10).grid(row=2, column=1, sticky="w", padx=8, pady=6)
        ttk.Label(settings_frame, text="Do dai moi canh (tham so cu, co the bo qua)").grid(
            row=2, column=2, sticky="w", padx=8, pady=6
        )
        ttk.Entry(settings_frame, textvariable=self.shot_len_var, width=10).grid(
            row=2, column=3, sticky="w", padx=8, pady=6
        )
        ttk.Checkbutton(settings_frame, text="Xao tron thu tu video nguon", variable=self.shuffle_var).grid(
            row=2, column=4, sticky="w", padx=8, pady=6
        )
        ttk.Label(settings_frame, text="Thiet bi xu ly video").grid(row=3, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            settings_frame,
            textvariable=self.device_var,
            values=("auto", "cuda", "cpu"),
            width=12,
            state="readonly",
        ).grid(row=3, column=1, sticky="w", padx=8, pady=6)
        ttk.Label(settings_frame, text="auto: uu tien GPU NVENC neu co").grid(
            row=3, column=2, columnspan=3, sticky="w", padx=8, pady=6
        )

        actions_frame = ttk.Frame(left_panel)
        actions_frame.pack(fill="x", pady=(0, 10))
        self.render_btn = ttk.Button(actions_frame, text="Bat dau tao video", command=self._start_batch_render)
        self.render_btn.pack(side="left")
        ttk.Button(actions_frame, text="Xem video moi nhat", command=self._open_latest_output_video).pack(
            side="left", padx=8
        )

        lists_frame = ttk.Frame(left_panel)
        lists_frame.pack(fill="both", expand=True)

        videos_frame = ttk.LabelFrame(lists_frame, text="Danh sach video nguon")
        videos_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.video_listbox = tk.Listbox(videos_frame, height=8)
        self.video_listbox.pack(fill="both", expand=True, padx=8, pady=8)
        video_actions = ttk.Frame(videos_frame)
        video_actions.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(video_actions, text="Lam moi", command=self._refresh_video_list).pack(side="left")
        ttk.Button(video_actions, text="Them video moi", command=self._import_videos).pack(side="left", padx=6)

        audios_frame = ttk.LabelFrame(lists_frame, text="Danh sach audio (moi file tao 1 video)")
        audios_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self.audio_listbox = tk.Listbox(audios_frame, height=8)
        self.audio_listbox.pack(fill="both", expand=True, padx=8, pady=8)
        audio_actions = ttk.Frame(audios_frame)
        audio_actions.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(audio_actions, text="Lam moi", command=self._refresh_audio_list).pack(side="left")
        ttk.Button(audio_actions, text="Them", command=self._import_audios).pack(side="left", padx=4)
        ttk.Button(audio_actions, text="Sua ten", command=self._rename_selected_audio).pack(side="left", padx=4)
        ttk.Button(audio_actions, text="Xoa", command=self._delete_selected_audio).pack(side="left", padx=4)

        outputs_frame = ttk.LabelFrame(right_panel, text="Video da render")
        outputs_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.output_listbox = tk.Listbox(outputs_frame, height=9)
        self.output_listbox.pack(fill="both", expand=True, padx=8, pady=8)
        self.output_listbox.bind("<Double-Button-1>", lambda _evt: self._open_selected_output_video())
        output_actions = ttk.Frame(outputs_frame)
        output_actions.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(output_actions, text="Lam moi", command=self._refresh_output_list).pack(side="left")
        ttk.Button(output_actions, text="Xem video da chon", command=self._open_selected_output_video).pack(
            side="left", padx=6
        )
        ttk.Button(output_actions, text="Mo thu muc output", command=self._open_output_dir).pack(side="left", padx=6)

        logs_frame = ttk.LabelFrame(right_panel, text="Nhat ky xu ly")
        logs_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(logs_frame, height=8, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

    def _choose_video_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(self.video_dir))
        if not selected:
            return
        self.video_dir = Path(selected)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.video_path_label.config(text=f"Thu muc video: {self.video_dir}")
        self._refresh_video_list()

    def _choose_audio_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(self.audio_dir))
        if not selected:
            return
        self.audio_dir = Path(selected)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.audio_path_label.config(text=f"Thu muc audio: {self.audio_dir}")
        self._refresh_audio_list()

    def _choose_output_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(self.output_dir))
        if not selected:
            return
        self.output_dir = Path(selected)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path_label.config(text=f"Thu muc xuat video: {self.output_dir}")
        self._refresh_output_list()

    def _refresh_video_list(self) -> None:
        self.video_files = list_video_files(self.video_dir)
        self.video_listbox.delete(0, tk.END)
        for path in self.video_files:
            self.video_listbox.insert(tk.END, path.name)

    def _refresh_audio_list(self) -> None:
        self.audio_files = list_audio_files(self.audio_dir)
        self.audio_listbox.delete(0, tk.END)
        for path in self.audio_files:
            self.audio_listbox.insert(tk.END, path.name)

    def _refresh_output_list(self) -> None:
        self.output_files = list_output_video_files(self.output_dir)
        self.output_listbox.delete(0, tk.END)
        for path in self.output_files:
            self.output_listbox.insert(tk.END, path.name)

    def _import_videos(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Chon file video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v"), ("All files", "*.*")],
        )
        if not selected:
            return
        imported = import_files_to_dir(list(selected), self.video_dir, SUPPORTED_VIDEO_EXTS)
        if not imported:
            messagebox.showwarning("Khong the them file", "Khong co file video hop le.")
            return
        self._refresh_video_list()
        self._append_log(f"Da them {len(imported)} file video.")

    def _import_audios(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Chon file audio",
            filetypes=[("Audio files", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus"), ("All files", "*.*")],
        )
        if not selected:
            return
        imported = import_files_to_dir(list(selected), self.audio_dir, SUPPORTED_AUDIO_EXTS)
        if not imported:
            messagebox.showwarning("Khong the them file", "Khong co file audio hop le.")
            return
        self._refresh_audio_list()
        self._append_log(f"Da them {len(imported)} file audio.")

    def _get_selected_audio_path(self) -> Path | None:
        selection = self.audio_listbox.curselection()
        if not selection:
            return None
        idx = int(selection[0])
        if idx < 0 or idx >= len(self.audio_files):
            return None
        return self.audio_files[idx]

    def _rename_selected_audio(self) -> None:
        src = self._get_selected_audio_path()
        if src is None:
            messagebox.showinfo("Thong bao", "Hay chon 1 file audio de sua.")
            return
        new_name = simpledialog.askstring("Sua ten audio", "Nhap ten moi:", initialvalue=src.name, parent=self.root)
        if new_name is None:
            return
        new_name = new_name.strip()
        if not new_name:
            messagebox.showerror("Loi", "Ten moi khong duoc de trong.")
            return
        if Path(new_name).suffix == "":
            new_name = f"{new_name}{src.suffix}"
        if Path(new_name).suffix.lower() not in SUPPORTED_AUDIO_EXTS:
            messagebox.showerror("Loi", f"Dinh dang audio khong hop le: {Path(new_name).suffix}")
            return
        dst = src.with_name(new_name)
        if dst.exists() and dst.resolve() != src.resolve():
            messagebox.showerror("Loi", "Da ton tai file audio cung ten.")
            return
        try:
            src.rename(dst)
        except Exception as exc:
            messagebox.showerror("Loi", f"Khong the sua ten audio: {exc}")
            return
        self._refresh_audio_list()
        self._append_log(f"Da sua ten audio: {src.name} -> {dst.name}")

    def _delete_selected_audio(self) -> None:
        src = self._get_selected_audio_path()
        if src is None:
            messagebox.showinfo("Thong bao", "Hay chon 1 file audio de xoa.")
            return
        confirm = messagebox.askyesno("Xoa audio", f"Ban co chac muon xoa '{src.name}' khong?")
        if not confirm:
            return
        try:
            src.unlink()
        except Exception as exc:
            messagebox.showerror("Loi", f"Khong the xoa audio: {exc}")
            return
        self._refresh_audio_list()
        self._append_log(f"Da xoa audio: {src.name}")

    def _get_selected_output_video(self) -> Path | None:
        selection = self.output_listbox.curselection()
        if not selection:
            return None
        idx = int(selection[0])
        if idx < 0 or idx >= len(self.output_files):
            return None
        return self.output_files[idx]

    def _open_selected_output_video(self) -> None:
        target = self._get_selected_output_video()
        if target is None:
            messagebox.showinfo("Thong bao", "Hay chon 1 video output de xem.")
            return
        try:
            open_with_default_app(target)
        except Exception as exc:
            messagebox.showerror("Loi mo video", str(exc))

    def _open_latest_output_video(self) -> None:
        self._refresh_output_list()
        if not self.output_files:
            messagebox.showinfo("Thong bao", "Chua co video output nao.")
            return
        latest = max(self.output_files, key=lambda p: p.stat().st_mtime)
        try:
            open_with_default_app(latest)
        except Exception as exc:
            messagebox.showerror("Loi mo video", str(exc))

    def _open_output_dir(self) -> None:
        try:
            open_with_default_app(self.output_dir)
        except Exception as exc:
            messagebox.showerror("Loi mo thu muc", str(exc))

    def _append_log(self, message: str) -> None:
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def _append_log_threadsafe(self, message: str) -> None:
        self.root.after(0, lambda: self._append_log(message))

    def _set_running(self, running: bool) -> None:
        self.is_running = running
        state = "disabled" if running else "normal"
        self.render_btn.config(state=state)

    def _start_batch_render(self) -> None:
        if self.is_running:
            return
        self._refresh_video_list()
        self._refresh_audio_list()

        if not self.video_files:
            messagebox.showerror("Thieu video", "Chua co file video nao.")
            return
        if not self.audio_files:
            messagebox.showerror("Thieu audio", "Chua co file audio nao.")
            return

        try:
            fps = int(self.fps_var.get().strip())
            if fps <= 0:
                raise ValueError("FPS phai > 0")
        except ValueError as exc:
            messagebox.showerror("Gia tri khong hop le", str(exc))
            return

        shot_len = 4.0
        size = (1920, 1080) if self.orientation_var.get() == "landscape" else (1080, 1920)
        render_mode = self.mode_var.get().strip() or "even"
        device_choice = self.device_var.get().strip().lower() or "auto"
        if device_choice not in {"auto", "cuda", "cpu"}:
            messagebox.showerror("Gia tri khong hop le", "Thiet bi xu ly phai la auto, cuda hoac cpu.")
            return

        self._set_running(True)
        worker = threading.Thread(
            target=self._run_batch_render,
            args=(render_mode, fps, shot_len, size, bool(self.shuffle_var.get()), device_choice),
            daemon=True,
        )
        worker.start()

    def _run_batch_render(
        self,
        render_mode: str,
        fps: int,
        shot_len: float,
        size: tuple[int, int],
        shuffle: bool,
        device_choice: str,
    ) -> None:
        try:
            total = len(self.audio_files)
            self._append_log_threadsafe(
                f"Bat dau tao {total} video voi kich thuoc {size[0]}x{size[1]} | device={device_choice}."
            )
            newest_output: Path | None = None
            for idx, audio_path in enumerate(self.audio_files, start=1):
                out_path = unique_destination_path(self.output_dir, f"{audio_path.stem}.mp4")
                self._append_log_threadsafe(f"[{idx}/{total}] {audio_path.name} -> {out_path.name}")
                args = argparse.Namespace(
                    audio=audio_path,
                    videos=self.video_dir,
                    out=out_path,
                    mode=render_mode,
                    device=device_choice,
                    shot_len=shot_len,
                    size=size,
                    fps=fps,
                    shuffle=shuffle,
                    watermark=None,
                    wm_scale=0.12,
                    wm_pos="bottom-right",
                    bgm=None,
                    bgm_volume=0.12,
                    bgm_duck_volume=0.035,
                )
                out_file, logs = execute_render(args)
                newest_output = out_file
                for line in logs:
                    self._append_log_threadsafe(f"  {line}")
                self._append_log_threadsafe(f"  Hoan tat: {out_file}")
                self.root.after(0, self._refresh_output_list)
            self._append_log_threadsafe("Da xu ly xong toan bo audio.")
            if newest_output is not None:
                self._append_log_threadsafe(f"Video moi nhat: {newest_output.name}. Bam 'Xem video moi nhat' de mo.")
        except Exception as exc:
            self._append_log_threadsafe(f"Loi: {exc}")
            self.root.after(0, lambda: messagebox.showerror("Render loi", str(exc)))
        finally:
            self.root.after(0, lambda: self._set_running(False))

def launch_gui() -> None:
    if tk is None or ttk is None or filedialog is None or messagebox is None or simpledialog is None:
        raise RuntimeError("Tkinter is not available. Install python with Tk support to use --gui.")
    root = tk.Tk()
    VideoMontageGUI(root=root, module_dir=Path(__file__).resolve().parent)
    root.mainloop()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    launch_gui_by_default = args.audio is None and args.videos is None and args.out is None
    if args.gui or launch_gui_by_default:
        try:
            launch_gui()
            return
        except Exception as exc:
            parser.exit(status=1, message=f"Error: {exc}\n")

    validate_args(args, parser)
    try:
        out_path, logs = execute_render(args)
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")
    for line in logs:
        print(line, file=sys.stderr)
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()

