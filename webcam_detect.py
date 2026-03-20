from ultralytics import YOLO
import cv2

# Load your trained YOLO model
model = YOLO("best.pt")   # path to your trained model

# IDs of only beauty-product classes from classes.txt
BEAUTY_CLASSES = [1, 3, 4, 5, 6, 7, 8]

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break

    # Run YOLO prediction on the frame, focusing only on beauty products
    results = model.predict(
        source=frame,
        conf=0.5,               # increase if still too many wrong boxes
        iou=0.45,
        imgsz=640,
        classes=BEAUTY_CLASSES, # only beauty products
        verbose=False
        # you can also try: half=True on GPU to speed up
    )

    # Draw predictions on the frame
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("YOLO Webcam Detection - Beauty Products Only", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
