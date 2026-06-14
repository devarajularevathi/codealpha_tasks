from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8s.pt")  # Better accuracy than yolov8n.pt

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("Object Detection Started")
print("Press 'Q' to Quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Run detection
    results = model(frame)

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Print detected objects
    detected_objects = set()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        object_name = model.names[cls_id]

        if object_name not in detected_objects:
            detected_objects.add(object_name)
            print("Detected:", object_name)

    # Display output
    cv2.imshow("Real-Time Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()