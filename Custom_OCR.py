import os
import cv2
import pytesseract
import numpy as np

# Set the Tesseract executable path (optional but recommended)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Dell\Desktop\Project_10\Dataset_OCR\Tesseract\doc"

# Path for input image
input_image_path = r"C:\Users\Dell\Desktop\Project_10\Dataset_OCR\thyrocare_0_36.jpg"
output_dir = r"C:\Users\Dell\Desktop\Project_10\out_csv"  # Directory to save cropped images and text files

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load YOLOv3 model and weights
yolo_config_path = r'C:\Users\Dell\Desktop\Project_10\yolov3.cfg'  # Path to YOLOv3 configuration file
yolo_weights_path = r'C:\Users\Dell\Desktop\Project_10\yolov3.weights'  # Path to YOLOv3 weights file
yolo_classes_path = r'C:\Users\Dell\Desktop\Project_10\coco.names'  # Path to COCO classes file

# Load YOLOv3 model
net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)

# Load COCO class labels (you may use your custom classes)
with open(yolo_classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layer names of the YOLO model
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load image
image = cv2.imread(input_image_path)
height, width = image.shape[:2]

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Perform forward pass to get bounding boxes
detections = net.forward(output_layers)

# List to store detected regions
detected_regions = []

# Analyze the YOLO detections
for output in detections:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Only consider objects with confidence > threshold (e.g., 0.5)
        if confidence > 0.5:
            # Get bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Save the detected bounding box (x, y, w, h)
            detected_regions.append((x, y, w, h))

            # Draw the bounding box and label on the image
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the image with bounding boxes
output_image_path = os.path.join(output_dir, 'image_with_boxes.jpg')
cv2.imwrite(output_image_path, image)

# Function to preprocess image for OCR
def preprocess_image_for_ocr(cropped_image):
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# Process each detected region, crop, preprocess and apply OCR
for i, (x, y, w, h) in enumerate(detected_regions):
    # Crop the detected region from the original image
    cropped_image = image[y:y+h, x:x+w]

    # Preprocess the cropped image for OCR
    preprocessed_image = preprocess_image_for_ocr(cropped_image)

    # Save the preprocessed cropped image
    cropped_image_path = os.path.join(output_dir, f'cropped_region_{i}.jpg')
    cv2.imwrite(cropped_image_path, preprocessed_image)

    # Use Tesseract to extract text from the preprocessed image
    extracted_text = pytesseract.image_to_string(preprocessed_image)

    # Save the extracted text to a file
    with open(os.path.join(output_dir, f'extracted_text_{i}.txt'), 'w') as text_file:
        text_file.write(extracted_text)

    print(f"Text extracted from region {i}:")
    print(extracted_text)

print(f"Processing completed. Check the '{output_dir}' directory for output files.")
