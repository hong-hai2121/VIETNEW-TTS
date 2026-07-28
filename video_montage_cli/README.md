# Video Montage CLI

CLI nho de ghep audio + folder video thanh 1 file MP4 bang FFmpeg.

## Cau truc

```text
video_montage_cli/
  main.py
  input/
    audio/
    videos/
  output/
```

## Cai FFmpeg

Can co ca `ffmpeg` va `ffprobe` trong `PATH`.

### Windows

1. Cai qua winget:
   - `winget install --id Gyan.FFmpeg -e`
2. Kiem tra:
   - `ffmpeg -version`
   - `ffprobe -version`

### macOS

1. Cai qua Homebrew:
   - `brew install ffmpeg`
2. Kiem tra:
   - `ffmpeg -version`
   - `ffprobe -version`

### Ubuntu/Debian

1. Cai:
   - `sudo apt update && sudo apt install -y ffmpeg`
2. Kiem tra:
   - `ffmpeg -version`
   - `ffprobe -version`

## Cach chay

### GUI (de xu ly nhieu audio)

```bash
python video_montage_cli/main.py --gui
```

Hoac chay khong tham so de mo GUI mac dinh:

```bash
python video_montage_cli/main.py
```

GUI ho tro:
- Load danh sach video tu thu muc dang chon.
- Import video moi vao thu muc video.
- Import nhieu file audio (moi file audio tao ra 1 file video rieng).
- Chon ti le khung hinh ngang `1920x1080` hoac doc `1080x1920`.
- Render batch ra thu muc `output`.

### CLI

Chay tu root repo:

```bash
python video_montage_cli/main.py \
  --audio video_montage_cli/input/audio.mp3 \
  --videos video_montage_cli/input/videos \
  --out video_montage_cli/output/final.mp4 \
  --mode speech \
  --size 1920x1080 \
  --fps 30 \
  --shuffle \
  --watermark assets/watermark.png \
  --wm-scale 0.12 \
  --wm-pos bottom-right \
  --bgm input/bgm.mp3
```

## Flags

- `--gui`: mo giao dien GUI de xu ly batch (khong can `--audio/--videos/--out`)
- `--audio`: file audio input (bat buoc neu chay CLI)
- `--videos`: folder chua cac video shot (bat buoc neu chay CLI)
- `--out`: file MP4 output (bat buoc neu chay CLI)
- `--mode`: `even` hoac `speech` (mac dinh `even`)
- `--shot-len`: tham so cu (giu lai de tuong thich), hien tai khong anh huong den mode `even`
- `--size`: kich thuoc output (vi du `1920x1080` hoac `1080x1920`)
- `--fps`: khung hinh/giay output
- `--shuffle`: tron thu tu video truoc khi build shot (mac dinh la theo ten file)
- `--watermark`: duong dan PNG watermark (optional)
- `--wm-scale`: ti le chieu ngang watermark so voi video (default `0.12`)
- `--wm-pos`: vi tri watermark: `top-left`, `top-right`, `bottom-left`, `bottom-right`
- `--bgm`: file background music de tron voi voice (optional)
- `--bgm-volume`: volume BGM khi khong co speech (default `0.12`)
- `--bgm-duck-volume`: volume BGM khi co speech (default `0.035`)

## Ghi chu mode

- `even`: 
  - Doc duration audio.
  - Duyet danh sach video (theo thu tu hoac shuffle).
  - Moi video se chay het clip roi moi chuyen sang video tiep theo.
  - Neu chi co 1 video thi lap lai video do den khi du thoi luong audio.
  - Resize + center-crop ve dung `--size`.
  - Gan audio va trim final bang dung duration audio.
- `speech`:
  - Dung `faster-whisper` de lay speech segments (`start/end`).
  - Chuan hoa do dai segment trong khoang `2.0s -> 6.5s` (segment dai se bi split).
  - Dung tung segment da chuan hoa lam do dai moi shot.
  - Neu whisper fail (thieu package, loi transcribe, khong co segment), tu dong fallback ve `even`.

## Watermark

- Watermark la PNG duoc overlay len video cuoi.
- Chieu ngang watermark scale theo chieu ngang video qua `--wm-scale` (thuong dung `0.10 -> 0.15`).
- Ho tro 4 vi tri qua `--wm-pos`: `top-left`, `top-right`, `bottom-left`, `bottom-right`.

## BGM + Ducking

- Neu co `--bgm`, tool se tron BGM voi voice.
- Neu lay duoc whisper segments, BGM se duck xuong `--bgm-duck-volume` khi voice dang noi.
- Neu khong co segments (hoac whisper fail), tool fallback sang BGM volume co dinh `--bgm-volume`.

## Cai them cho mode speech

Neu muon dung `--mode speech`, cai them:

```bash
pip install faster-whisper
```
