from ultralytics import YOLO
import cv2

# Load YOLOv8 model (downloads automatically first time)
model = YOLO("yolov8n.pt")  # nano version (fast)

# Load image
image_path = "test.jpeg"  # change to your image path
image = cv2.imread(image_path)

# Run detection
results = model(image)

# Draw bounding boxes
annotated_frame = results[0].plot()

# Save result
cv2.imwrite("output.jpg", annotated_frame)

# Show result
cv2.imshow("YOLO Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()