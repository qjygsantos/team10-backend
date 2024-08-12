import os
import json
import cv2
import requests
from flask import Flask, request, jsonify, render_template
from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image, ImageDraw, ImageFont
import difflib
import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Ensure the necessary directories exist
for directory in ['static/objects', 'static/detected_images']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Set environment variables for credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/potent-bloom-422217-a8-8b6e616ee921.json"  # For Google Cloud Vision API

app = Flask(__name__)

# Initialize Firebase Admin
cred = credentials.Certificate("/etc/secrets/psykitz-891d8-firebase-adminsdk-l7okt-38b1a73888.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'psykitz-891d8.appspot.com'})

db = firestore.client()
bucket = storage.bucket()

# Define predefined commands and symbols
predefined_commands = [
    "move forward (5)", "move forward", "move forward (2)",
    "move backward (5)", "move backward", "move backward (2)",
    "turn left", "turn left (2)", "turn left (5)", "turn right",
    "turn right (2)", "turn left (5)", "turn 180", "stop", "drive forward",
    "turn on light", "turn off light", "play sound", "reverse"
]

predefined_conditions = [
    "if obstacle detected", "if line detected", "if no light",
    "start", "end"
]

class InferenceClient:
    def __init__(self, api_url, api_key, model_id):
        self.api_url = api_url
        self.api_key = api_key
        self.model_id = model_id

    def detect_handwriting(self, data):
        # Initialize Google Cloud Vision Client
        client = vision.ImageAnnotatorClient()
        with open(data, 'rb') as image_file:
            content = image_file.read()
        image = types.Image(content=content)
        response = client.document_text_detection(image=image)
        texts = response.text_annotations
        if texts:
            return texts[0].description
        else:
            return "No text detected"

    def detect_diagram(self, image_path):
        image = cv2.imread(image_path)
        custom_configuration = InferenceConfiguration(confidence_threshold=0.5, iou_threshold=0.5)
        detection_client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)
        detection_client.configure(custom_configuration)
        detection_result_objects = detection_client.infer(image, model_id=self.model_id)

        detection_result = []
        boxes = []
        confidences = []
        
        for idx, prediction in enumerate(detection_result_objects["predictions"]):
            x = int(prediction["x"])
            y = int(prediction["y"])
            width = int(prediction["width"])
            height = int(prediction["height"])
            symbol_class = prediction["class"]
            confidence = prediction["confidence"]

            x1 = x - width // 2
            y1 = y - height // 2
            x2 = x + width // 2
            y2 = y + height // 2

            roi = image[y1:y2, x1:x2]
            roi_filename = f'cropped_image_{idx}.jpg'
            roi_path = os.path.join('static/objects', roi_filename)
            cv2.imwrite(roi_path, roi)

            text = self.perform_ocr(roi_path)

            matched_command = None
            if text != "No text detected" and symbol_class.lower() not in ['arrow', 'arrowhead']:
                matched_command = self.match_text_with_commands(text)
                
            # Store bounding boxes and confidences before applying NMS
            boxes.append([x1, y1, width, height])
            confidences.append(confidence)

            pos = y2 if symbol_class.lower().replace("rotation", "") != 'decision' else y1
            
            detection_with_ocr = {
                'type': symbol_class.lower().replace("rotation", ""),
                'coordinates': (x, y),
                'command': matched_command if text != "No text detected" else "",
                'width': width,
                'height': height,
                'pos': pos
            }
            detection_result.append(detection_with_ocr)
        # Apply NMS 
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.5)

        # Make sure indices are correct
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_results = [detection_result[i] for i in indices]
        else:
            filtered_results = detection_result
        
        #sort results   
        filtered_results.sort(key=lambda x: x["pos"])
        
        for i in range(1, len(filtered_results)):
            if filtered_results[i]['type'] == 'arrow' and filtered_results[i-1]['type'] == 'arrowhead':
                  filtered_results[i], filtered_results[i-1] = filtered_results[i-1], filtered_results[i]

        for idx, detection in enumerate(filtered_results):
            # Assign ID
            detection["id"] = idx + 1

        return filtered_results

    def perform_ocr(self, output_image_path):
        return self.detect_handwriting(output_image_path)

    def match_text_with_commands(self, text):
        normalized_text = text.strip().lower()
        if normalized_text == "no text detected":
            return None

        for command in predefined_commands:
            if command in normalized_text:
                return command

        for condition in predefined_conditions:
            if condition in normalized_text:
                return condition

        closest_match = difflib.get_close_matches(normalized_text, predefined_commands + predefined_conditions, n=1, cutoff=0.6)
        if closest_match:
            return closest_match[0]
        else:
            return None

    def print_result_with_ocr(self, detection_result, image_path):
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        #base scale for font
        base_scale = 0.0011  

        print("Inference Results with OCR:")
        for detection in detection_result:
            print(detection)
            x1 = int(detection["coordinates"][0] - detection["width"] // 2)
            y1 = int(detection["coordinates"][1] - detection["height"] // 2)
            x2 = int(detection["coordinates"][0] + detection["width"] // 2)
            y2 = int(detection["coordinates"][1] + detection["height"] // 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
            label = f"{detection['id']}. {detection['type']}"
            if detection['command']:
                label += f" ({detection['command']})"

            # Calculate font scale based on image dimensions
            font_scale = base_scale * max(image_width, image_height)
            thickness = max(1, int(font_scale * 2))  # Adjust thickness based on font scale

            # Draw text on the image
            cv2.putText(image, label, (x1 - 20, y1 + 5), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (192, 15, 252), thickness)

        output_image_path = os.path.join('static/detected_images', os.path.basename(image_path))
        cv2.imwrite(output_image_path, image)
        return output_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Save the uploaded image to a temporary path
        image_path = os.path.join('static/objects', file.filename)
        file.save(image_path)

        # Initialize OCR client
        OCR_CLIENT = InferenceClient(
            api_url="https://detect.roboflow.com",
            api_key="A6HQefLyBwFRsvEb8Adr",
            model_id="handwritten-flowchart-part-3/15"
        )
        
        # Perform detection
        detection_result = OCR_CLIENT.detect_diagram(image_path)

        # Save the image with bounding boxes
        output_image_path = OCR_CLIENT.print_result_with_ocr(detection_result, image_path)

        # Upload processed image to Firebase Storage
        blob = bucket.blob(f'detected_images/{os.path.basename(output_image_path)}')
        blob.upload_from_filename(output_image_path)
        image_url = blob.generate_signed_url(expiration=datetime.timedelta(days=7))

        # Save JSON results
        generated_code_path = os.path.join('static/detected_images', file.filename.split('.')[0] + '.json')
        with open(generated_code_path, 'w') as generated_code_file:
            json.dump(detection_result, generated_code_file, indent=4)

        # Upload JSON to Firebase Storage
        generated_code_blob = bucket.blob(f'detected_images/{os.path.basename(generated_code_path)}')
        generated_code_blob.upload_from_filename(generated_code_path)
        generated_code_url = ''

        # Save URLs to Firestore
        doc_ref = db.collection('image_data').document(file.filename.split('.')[0])
        doc_ref.set({
            'image_url': image_url,
            'generated_code_url': generated_code_url
        })

        # Clean up temporary files
        os.remove(image_path)
        os.remove(output_image_path)
        os.remove(generated_code_path)

        return jsonify({
            "message": "File processed successfully",
            "image_url": image_url,
            "generated_code_url": generated_code_url
        })

if __name__ == '__main__':
    app.run(debug=True)
