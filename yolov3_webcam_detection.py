

import cv2
import numpy as np

def load_data_and_preprocess(frame):
    """
    Preprocess the webcam frame for YOLO input.
    
    Args:
        frame (numpy.ndarray): The frame from the webcam.

    Returns:
        tuple: (original image, preprocessed image, height, width)
    """
    h, w = frame.shape[:2]
    pre_processed_image = cv2.dnn.blobFromImage(
        frame, scalefactor=1/255.0,  # Normalize pixel values
        size=(416, 416),  # Resize image to 416x416 (YOLO input size)
        swapRB=True,  # Convert BGR to RGB
        crop=False  # do not crop
    )
    return frame, pre_processed_image, h, w

def read_models_and_labels(LABLES_ADDRESS, NETWORK_WEIGHTS_ADDRESS, MODEL_CONFIGURATION_ADDRESS):
    """
    Loads class labels, YOLO model weights, and network configuration.
    
    Args:
        LABLES_ADDRESS (str): Path to the class labels file (.names).
        NETWORK_WEIGHTS_ADDRESS (str): Path to the YOLO weights file (.weights).
        MODEL_CONFIGURATION_ADDRESS (str): Path to the YOLO configuration file (.cfg).

    Returns:
        tuple: (list of labels, YOLO neural network model)
    """
    labels = open(LABLES_ADDRESS).read().strip().split("\n")  # Read labels
    net = cv2.dnn.readNet(NETWORK_WEIGHTS_ADDRESS, MODEL_CONFIGURATION_ADDRESS)  # Load the YOLO model
    return labels, net

def forwarding_pretrained_network(pre_processed_image, net):
    """
    Performs forward propagation through the YOLO network to get predictions.
    
    Args:
        pre_processed_image (numpy.ndarray): Preprocessed image ready for YOLO.
        net (cv2.dnn_Net): Preloaded YOLO neural network.

    Returns:
        list: Predictions from the YOLO model.
    """
    net.setInput(pre_processed_image)  # Set the input image for the network
    output_layers = ["yolo_82", "yolo_94", "yolo_106"]  # YOLOv3 has three output layers
    predictions = net.forward(output_layers)  # Get network predictions
    return predictions

def post_processing(predictions, w, h):
    """
    Processes the YOLO network output to extract bounding boxes, confidence scores, and class IDs.
    
    Args:
        predictions (list): Network output containing detected objects.
        w (int): Original image width.
        h (int): Original image height.

    Returns:
        tuple: (list of class IDs, list of confidence scores, list of bounding boxes)
    """
    classIDs = []  # List to store detected class IDs
    confidences = []  # List to store confidence scores
    boxes = []  # List to store bounding boxes

    # Iterate through each output layer
    for layer in predictions:
        for bounding_box in layer:  # Each bounding box contains 85 values in yolo v3
            scores = bounding_box[5:]  # Extract class scores (last 80 values)
            classID = np.argmax(scores)  # Get the class with the highest probability
            confidence = scores[classID]  # Get the confidence of the detected class

            if confidence > 0.3:  # Apply confidence threshold (30%)
                box = bounding_box[0:4] * np.array([w, h, w, h])  # Scale bounding box to original image size
                (centerX, centerY, width, height) = box.astype("int")  # Convert to integers
                x = int(centerX - width / 2)  # Get top-left x-coordinate for use opencv to show images results
                y = int(centerY - height / 2)  # Get top-left y-coordinate

                classIDs.append(classID)
                confidences.append(confidence)
                boxes.append([x, y, int(width), int(height)])

    return classIDs, confidences, boxes

def show_result(frame, classIDs, confidences, boxes, labels):
    """
    Draws bounding boxes and labels on the frame based on detected objects.
    
    Args:
        frame (numpy.ndarray): The frame from the webcam.
        classIDs (list): List of detected class IDs.
        confidences (list): List of confidence scores.
        boxes (list): List of bounding boxes (x, y, width, height).
        labels (list): List of object class labels.
    
    Displays:
        - The frame with bounding boxes and labels.
    """
    # Apply Non-Maximum Suppression (NMS) to remove redundant overlapping boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.5)  # 0.3 = confidence threshold, 0.5 = IOU threshold

    for i in idxs.flatten():  # Flatten the indices returned by NMS because it is a 2D array
        x, y, w, h = boxes[i]  # Extract bounding box coordinates

        # Draw a rectangle around the detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Green box

        # Create label text with class name and confidence score
        text = "{}: {:.2f}".format(labels[classIDs[i]], confidences[i])

        # Put label text above the bounding box
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 125, 0), 2)  

    # Display the frame with detections
    cv2.imshow("Webcam Detection", frame)

def detect_objects_in_webcam():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
    

    # Load YOLO model and labels
    labels, net = read_models_and_labels(
        ".\yolo files\coco.names",  # Path to class labels
        ".\yolo files\yolov3.weights",  # Path to YOLOv3 weights
        ".\yolo files\yolov3.cfg"  # Path to YOLOv3 configuration file
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Load and preprocess the frame
        img, pre_processed_image, h, w = load_data_and_preprocess(frame)

        # Perform forward pass through YOLO network
        predictions = forwarding_pretrained_network(pre_processed_image, net)

        # Process predictions and extract bounding boxes
        classIDs, confidences, boxes = post_processing(predictions, w, h)

        # Display results on the frame
        show_result(frame, classIDs, confidences, boxes, labels)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


print("*** press 'q' to close Webcam ***")
detect_objects_in_webcam()
