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
    "move forward", "move backward", "turn left", "drive forward", "drive backward", "turn right", "spin",
    "stop", "turn on light", "turn off light", "play sound", "repeat"
]

predefined_conditions = [
    "if obstacle ahead", "if no obstacle", "if light detected", "if no light",
    "start", "end", "if touch sensor pressed"
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
        custom_configuration = InferenceConfiguration(confidence_threshold=0.5)
        detection_client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)
        detection_client.configure(custom_configuration)
        detection_result_objects = detection_client.infer(image, model_id=self.model_id)

        detection_result = []
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

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            roi = image[y1:y2, x1:x2]
            roi_filename = f'cropped_image_{idx}.jpg'
            roi_path = os.path.join('static/objects', roi_filename)
            cv2.imwrite(roi_path, roi)

            text = self.perform_ocr(roi_path)

            matched_command = None
            if text != "No text detected" and symbol_class.lower() not in ['arrow', 'arrowhead']:
                matched_command = self.match_text_with_commands(text)

            detection_with_ocr = {
                'type': symbol_class.lower().replace("rotation", ""),
                'coordinates': (x, y),
                'text': text if text != "No text detected" else "",
                'width': width,
                'height': height,
                'command': matched_command
            }
            detection_result.append(detection_with_ocr)

        detection_result.sort(key=lambda x: x["coordinates"][1])

        for idx, detection in enumerate(detection_result):
            detection['id'] = idx + 1

        return detection_result

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
                label += f" - {detection['command']}"
    
            # Adjust the font type and size for a friendlier look
            font = cv2.FONT_HERSHEY_TRIPLEX  # Simplex is clear and readable
            font_scale = 4  # Slightly larger font size for readability
            font_color = (192, 15, 252)  # Light blue color in BGR
            font_thickness = 2
    
            # Calculate the text size
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_width = text_size[0]
    
            text_x = x1 - text_width  # Move the text to the left of the bounding box
            text_y = y1 + text_size[1] + 15 
    
            # Put text on the image with the updated font settings
            cv2.putText(image, label, (text_x, text_y), font, font_scale, font_color, font_thickness)
    
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
