from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")  # keep your existing best model

# IDs of only beauty-product classes from classes.txt
BEAUTY_CLASSES = [1, 3, 4, 5, 6, 7, 8]

# Run prediction on an image
results = model.predict(
    source="test.jpg",      # change to your image path
    conf=0.5,               # higher confidence -> fewer false positives
    iou=0.45,               # NMS IoU
    imgsz=640,              # standard YOLO size
    classes=BEAUTY_CLASSES, # IGNORE airplane & car
    verbose=False
)

# Show results
results[0].show()
