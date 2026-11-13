# 🎬 Sample Videos

Place test construction site videos in this folder for processing by StructureGuard.

## Recommended Test Videos

### Option 1: Roboflow / YouTube
Search YouTube for "construction site safety footage" or use clips from:
- Roboflow Construction Safety Dataset sample videos
- OSHA training footage (public domain)

### Option 2: Webcam
Run StructureGuard with `--source 0` to use your webcam for live testing.

### Option 3: Synthetic Test
Any video with people walking will work for basic tracking demonstration (the COCO-trained model detects the `person` class). PPE violation detection requires custom-trained weights.

## Supported Formats
- `.mp4` (recommended)
- `.avi`
- `.mov`
- `.mkv`
- Any format supported by OpenCV's VideoCapture

## Usage
```bash
python main.py --source sample_videos/your_video.mp4 --save-video
```

## File Placement
```
sample_videos/
├── README_VIDEOS.md    ← (this file)
├── site.mp4            ← your test video
└── demo_clip.avi       ← additional test files
```
