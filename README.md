# YOLO Beauty Products Detection (Label Studio Dataset)

This project uses your own captured images and Label Studio annotations to train and run a YOLO object detection model on beauty products.

## Project files
- `best.pt`: trained model weights.
- `detect.py`: run inference on an image.
- `webcam_detect.py`: run realtime webcam inference.
- `YoloModelScrip.ipynb`: training/experiment notebook.
- `BeautyProducts images/`: dataset image files + notes.
- `README.md`: this document.

## Dataset creation workflow (your setup)
1. Capture images manually (camera/phone) and copy into the dataset folder.
2. Annotate objects with Label Studio (bounding boxes + class labels).
3. Export labels from Label Studio.
4. Convert Label Studio export to YOLO format:
   - one `.txt` file per image
   - each row: `class x_center y_center width height` normalized (0..1)
5. Arrange directories:
   - `images/train/`, `images/val/` (or `BeautyProducts images/`)
   - `labels/train/`, `labels/val/`

## data.yaml example
```yaml
train: "images/train"
val: "images/val"
nc: 8
names: ['class0','class1','class2','class3','class4','class5','class6','class7']
```

Adjust `nc` and class names to your dataset.

## Environment setup
```cmd
cd d:\YOLO
python -m pip install ultralytics opencv-python
```

## Training using Ultralytics YOLO
```cmd
yolo detect train data=data.yaml model=yolov8n.pt epochs=50
```

## Running inference
### Image inference
1. Replace `source="test.jpg"` in `detect.py` with your image path.
2. Run:
```cmd
python detect.py
```

### Webcam inference
```cmd
python webcam_detect.py
```
Press `q` to quit.

## `BEAUTY_CLASSES` filter
In both scripts, this filter is enabled:
```python
BEAUTY_CLASSES = [1, 3, 4, 5, 6, 7, 8]
```
Change IDs to match your `classes.txt` mapping.

## Notes
- Keep dataset and labels in sync.
- Keep the notebook for workflow reproducibility.


