# 📦 Model Weights

## Default Model
StructureGuard uses **YOLOv8n** (nano) as the default detection backbone. On first run, the `ultralytics` library auto-downloads `yolov8n.pt` — no manual action needed.

## Custom PPE-Trained Weights (Recommended)
For accurate PPE detection (helmets, vests, harnesses, gloves), replace the default COCO-trained weights with a model fine-tuned on a construction safety dataset:

### Option 1: Roboflow Construction Safety Dataset
1. Visit: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
2. Download the dataset in **YOLOv8 format**
3. Train using Ultralytics:
   ```bash
   yolo detect train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
   ```
4. Copy `best.pt` to this folder
5. Update `config.yaml`:
   ```yaml
   model:
     weights: "models/best.pt"
   ```

### Option 2: Pre-Trained PPE Models
Search Roboflow Universe for pre-trained PPE detection models:
- https://universe.roboflow.com/search?q=ppe+detection

### Class Mapping
When using custom weights, ensure your model's classes match the `classes` list in `config.yaml`:
```
0: person
1: helmet
2: no_helmet
3: safety_vest
4: no_vest
5: harness
6: no_harness
7: gloves
8: machinery
9: scaffold
```

## File Placement
Place your `.pt` weight files in this directory:
```
models/
├── README_MODELS.md    ← (this file)
├── yolov8n.pt          ← auto-downloaded on first run
└── best.pt             ← your custom-trained weights
```
