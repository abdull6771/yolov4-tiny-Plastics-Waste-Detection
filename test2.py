import cv2
import numpy as np

# Load YOLOv4-tiny model
model_weights = "yolov4-tiny-custom_best.weights"
model_config = "yolov4-tiny-custom.cfg"
class_labels = "obj.names"  # Replace with the appropriate class labels file

net = cv2.dnn.readNet(model_weights, model_config)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# Load class labels
with open(class_labels, 'r') as f:
    classes = f.read().strip().split('\n')

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
#output_layers=[]
#for i in net.getUnconnectedOutLayers():
#    output_layers.append(layer_names[i-1])
# Open a connection to the webcam (usually 0 or 1)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)

    # Set the input to the neural network
    net.setInput(blob)

    # Perform forward pass and get predictions
    outs = net.forward(output_layers)

    # Post-process the detections
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Adjust confidence threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left and bottom-right coordinates of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Draw the bounding boxes and labels on the frame
    for i in indices:
        i = i
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display class label and confidence
        text = f"{label} {confidence:.2f}"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with detections
    cv2.imshow("YOLOv4-tiny Real-Time", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
