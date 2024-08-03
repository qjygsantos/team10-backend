from flask import Flask, request, jsonify, send_file, render_template
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import requests
import difflib
import cv2
import os
from google.cloud import vision
from google.cloud.vision_v1 import types
import difflib
from PIL import Image, ImageDraw, ImageFont
import json
from difflib import get_close_matches  

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/potent-bloom-422217-a8-8b6e616ee921.json"

app = Flask(__name__)


# Define predefined commands and symbols
predefined_commands = [
    "move forward (slow)", "move forward (normal)", "move forward (fast)",
    "move backward (slow)", "move backward (normal)", "move backward (fast)",
    "turn left", "drive forward", "turn right", "spin", "make sound", "stop",
    "turn on light", "turn off light", "wait for", "play sound", "repeat"
]

predefined_conditions = [
    "if obstacle ahead", "if no obstacle", "if light detected", "if no light",
    "start", "end"
]

class InferenceClient:
    def __init__(self, api_url, api_key, model_id):
        self.api_url = api_url
        self.api_key = api_key
        self.model_id = model_id

    def detect_handwriting(self, data):
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
            output_image_path = f'static/objects/cropped_image_{idx}.jpg'
            cv2.imwrite(output_image_path, roi)

            text = self.perform_ocr(output_image_path)

            matched_command = None
            if text != "No text detected" and symbol_class.lower() not in ['arrow', 'arrowhead']:
                matched_command = self.match_text_with_commands(text)

            detection_with_ocr = {
                'type': symbol_class.lower().replace("rotation", ""),
                'coordinates': (x, y),
                'text': matched_command if text != "No text detected" else "",
                'width': width,
                'height': height

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
        #upload and save input image to path
        image_path = os.path.join('static/detected_images', file.filename)
        file.save(image_path)
        
        #instantiate
        OCR_CLIENT = InferenceClient(
            api_url="https://detect.roboflow.com",
            api_key="A6HQefLyBwFRsvEb8Adr",
            model_id = "handwritten-flowchart-part-3/15"
        )
        
        #OCR API keyyy

        #perform detection
        detection_result = OCR_CLIENT.detect_diagram(image_path)

        # Save the image with bounding boxes
        output_image_path = draw_bounding_boxes(image_path, detection_result)

        # Save JSON results
        json_output_path = os.path.join('static/detected_images', file.filename.split('.')[0] + '.json')
        with open(json_output_path, 'w') as json_file:
            json.dump(detection_result, json_file, indent=4)
         
         
        # Clean up temp images 
        objects_folder = 'static/objects'
        for filename in os.listdir(objects_folder):
            file_path = os.path.join(objects_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        return jsonify({
            "message": "File processed successfully",
            "image_url": output_image_path,
            "json_url": json_output_path
        })

def draw_bounding_boxes(image_path, detections):
    #take image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    font_size = 12  # Increase this value for a larger font
    font = ImageFont.truetype("arial.ttf", font_size)  # Adjust font and size as needed
    
    #put bounding box
    for detection in detections:
        x, y = detection['coordinates']
        width = detection['width']
        height = detection['height']
        box = [(x - width // 2, y - height // 2), (x + width // 2, y + height // 2)]

        draw.rectangle(box, outline="red", width=2)

        text = detection['text'] if detection['text'] else ''
        symbol_type = detection['type']

        # Draw text beside the symbol
        text_to_draw = f"{symbol_type}: {text}"
        text_position = (x - width // 2 - 10, y - height // 2)  # Move text left and adjust vertical position
        draw.text(text_position, text_to_draw, font=font, fill="blue")
            
    #save image with bounding box
    output_image_path = os.path.join('static/detected_images', os.path.basename(image_path))
    image.save(output_image_path) 
    return output_image_path

if __name__ == '__main__':
    app.run(debug=True)

















