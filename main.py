import cv2
from ultralytics import YOLO

# Path to your trained model
model_path = r"C:\Users\charlesmata\Downloads\best (4).pt"  # Use raw string (r"") to handle backslashes in the file path
model = YOLO(model_path)  # Load your trained YOLO model

# Initialize webcam (use 0 for the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop for real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform detection
    results = model.predict(source=frame, conf=0.5, save=False, show=False)

    # Visualize the results on the frame
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        label = f"{model.names[int(class_id)]}: {score:.2f}"

        # Draw bounding box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Sign Language Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

