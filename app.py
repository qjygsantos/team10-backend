import os
import json
import cv2
import requests
from flask import Flask, request, jsonify, render_template
from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image, ImageDraw, ImageFont
import difflib
import firebase_admin
from firebase_admin import credentials, firestore, storage
from inference_sdk import InferenceHTTPClient, InferenceConfiguration


# Ensure the temp directory exists
if not os.path.exists('temp'):
    os.makedirs('temp')

# Set environment variables for credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/potent-bloom-422217-a8-8b6e616ee921.json"  # For Google Cloud Vision API
os.environ["GOOGLE_APPLICATION_CREDENTIALS_FIREBASE"] = "/etc/secrets/psykitz-891d8-firebase-adminsdk-l7okt-38b1a73888.json"  # For Firebase

app = Flask(__name__)

# Initialize Firebase Admin
firebase_cred = credentials.Certificate(os.environ["GOOGLE_APPLICATION_CREDENTIALS_FIREBASE"])
firebase_admin.initialize_app(firebase_cred, {
    'storageBucket': 'psykitz-891d8.appspot.com'
})
db = firestore.Client()
bucket = storage.bucket()

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

            roi = image[y1:y2, x1:x2]
            roi_filename = f'cropped_image_{idx}.jpg'
            roi_path = f'temp/{roi_filename}'
            cv2.imwrite(roi_path, roi)

            # Upload cropped image to Firebase Storage
            blob = bucket.blob(f'objects/{roi_filename}')
            blob.upload_from_filename(roi_path)

            # Generate URL for the uploaded image
            roi_url = blob.public_url

            text = self.perform_ocr(roi_path)

            matched_command = None
            if text != "No text detected" and symbol_class.lower() not in ['arrow', 'arrowhead']:
                matched_command = self.match_text_with_commands(text)

            detection_with_ocr = {
                'type': symbol_class.lower().replace("rotation", ""),
                'coordinates': (x, y),
                'text': matched_command if text != "No text detected" else "",
                'width': width,
                'height': height,
                'image_url': roi_url
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
        image_path = os.path.join('temp', file.filename)
        file.save(image_path)

        # Initialize OCR client
        OCR_CLIENT = InferenceClient(
            api_url="https://detect.roboflow.com",
            api_key="A6HQefLyBwFRsvEb8Adr",
            model_id = "handwritten-flowchart-part-3/15"
        )
        
        # Perform detection
        detection_result = OCR_CLIENT.detect_diagram(image_path)

        # Save the image with bounding boxes
        output_image_path = draw_bounding_boxes(image_path, detection_result)

        # Upload processed image to Firebase Storage
        blob = bucket.blob(f'detected_images/{os.path.basename(output_image_path)}')
        blob.upload_from_filename(output_image_path)
        image_url = blob.public_url

        # Save JSON results
        json_output_path = os.path.join('temp', file.filename.split('.')[0] + '.json')
        with open(json_output_path, 'w') as json_file:
            json.dump(detection_result, json_file, indent=4)

        # Upload JSON to Firebase Storage
        json_blob = bucket.blob(f'detected_images/{os.path.basename(json_output_path)}')
        json_blob.upload_from_filename(json_output_path)
        json_url = json_blob.public_url
        
        # Save URLs to Firestore
        doc_ref = db.collection('image_data').document(file.filename.split('.')[0])
        doc_ref.set({
            'image_url': image_url,
            'json_url': json_url
        })

        # Clean up temporary files
        os.remove(image_path)
        os.remove(output_image_path)
        os.remove(json_output_path)

        return jsonify({
            "message": "File processed successfully",
            "image_url": image_url,
            "json_url": json_url
        })

def draw_bounding_boxes(image_path, detections):
    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    font_size = 12
    font = ImageFont.load_default()
    
    for detection in detections:
        x, y = detection['coordinates']
        width = detection['width']
        height = detection['height']
        box = [(x - width // 2, y - height // 2), (x + width // 2, y + height // 2)]

        draw.rectangle(box, outline="red", width=2)

        text = detection['text'] if detection['text'] else ''
        symbol_type = detection['type']

        text_to_draw = f"{symbol_type}: {text}"
        text_position = (x - width // 2 - 10, y - height // 2)
        draw.text(text_position, text_to_draw, font=font, fill="blue")
            
    output_image_path = os.path.join('temp', os.path.basename(image_path))
    image.save(output_image_path)
    return output_image_path

if __name__ == '__main__':
    app.run(debug=True)
