import os
import json
import cv2
import requests
from flask import Flask, request, jsonify, render_template
from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image, ImageDraw, ImageFont
import difflib
from difflib import get_close_matches
from skimage.filters import threshold_local
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
    "turn on light", "turn off light", "play sound", "reverse", "start", "end"
]

predefined_conditions = [
    "obstacle not detected", "line not detected", "touch sensor not pressed", "if i <= 2", "if i <= 3", "if i <= 5", "if i <= 10"
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

            if symbol_class.lower().replace("rotation", "") == 'decision':
                pos = y1 + 5

            elif symbol_class.lower().replace("rotation", "") == 'arrow':
                pos = y2 - 10

            elif symbol_class.lower().replace("rotation", "") == 'terminator' and matched_command == 'end':
                pos = y2 + 10

            else:
                pos = y2

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
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.6)

        # Make sure indices are correct
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_results = [detection_result[i] for i in indices]
        else:
            filtered_results = detection_result
         
        # Sort results
        filtered_results.sort(key=lambda x: x["pos"])
        for i in range(len(filtered_results) - 1):
            #WHILE Implementation
            if filtered_results[i]['type'] == 'decision' and filtered_results[i + 1]['type'] == 'arrowhead' and \
            filtered_results[i]['command'] in ["obstacle not detected", "line not detected","touch sensor not pressed"]:
                # Remove the second arrowhead
                removed_arrowhead = filtered_results.pop(i + 1)
                # Insert it five positions ahead if possible
                new_index = i + 4
                if new_index < len(filtered_results):
                    filtered_results.insert(new_index, removed_arrowhead)

            #DO-WHILE Implementation
            if i > 0 and i + 1 < len(filtered_results) and \
                          filtered_results[i]['type'] == 'arrowhead' and \
                          filtered_results[i + 1]['type'] == 'process' and \
                          filtered_results[i - 1]['type'] == 'arrowhead':
                    removed_arrowhead = filtered_results.pop(i)

                    # Insert it five positions ahead if possible
                    new_index = i + 5
                    if new_index < len(filtered_results):
                        filtered_results.insert(new_index, removed_arrowhead)

            #FOR LOOP Implementation
            if filtered_results[i]['type'] == 'decision' and filtered_results[i + 1]['type'] == 'arrowhead' and \
            filtered_results[i]['command'] in ["if i <= 2", "if i <= 3", "if i <= 5", "if i <= 10"]:
                #find the next arrow element with width > 100
                j = i + 1
                n = len(filtered_results)
                while j < n and (filtered_results[j]['type'] != 'arrow' or filtered_results[j]['height'] <= 300):
                    j += 1
                # Remove the second arrowhead
                removed_arrowhead = filtered_results.pop(i + 1)
                # Insert it five positions ahead if possible
                new_index = j
                if new_index < len(filtered_results):
                    filtered_results.insert(new_index, removed_arrowhead)

        for idx, detection in enumerate(filtered_results):
            # Assign ID
            detection["order"] = idx + 1

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

        closest_match = difflib.get_close_matches(normalized_text, predefined_commands + predefined_conditions, n=1, cutoff=0.2)
        if closest_match:
            return closest_match[0]
        else:
            return None

    def print_result_with_ocr(self, detection_result, image_path):
            image = cv2.imread(image_path)
            image_height, image_width = image.shape[:2]

            # Base scale for text
            base_scale = 0.00075  # Experiment with this value as needed

            print("Inference Results with OCR:")
            for detection in detection_result:
                print(detection)

                x1 = int(detection["coordinates"][0] - detection["width"] // 2)
                y1 = int(detection["coordinates"][1] - detection["height"] // 2)
                x2 = int(detection["coordinates"][0] + detection["width"] // 2)
                y2 = int(detection["coordinates"][1] + detection["height"] // 2)

                if detection["type"] == "arrow":
                    x = x1
                    y = y1

                if detection["type"] == "arrowhead":
                    x = x2
                    y = y2

                else:
                    x = detection["coordinates"][0]
                    y = detection["coordinates"][1]

                x1 = int(detection["coordinates"][0] - detection["width"] // 2)
                y1 = int(detection["coordinates"][1] - detection["height"] // 2)
                x2 = int(detection["coordinates"][0] + detection["width"] // 2)
                y2 = int(detection["coordinates"][1] + detection["height"] // 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{detection['order']}. {detection['type']}"
                if detection['command']:
                    label += f" ({detection['command']})"

                # Calculate font scale based on image dimensions
                font_scale = base_scale * max(image_width, image_height)
                thickness = max(1, int(font_scale * 2))  # Adjust thickness based on font scale

                # Draw text on the image
                cv2.putText(image, label, (x1 - 20, y1 + 5), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 255), thickness)

            output_image_path = os.path.join('static/detected_images', os.path.basename(image_path))
            cv2.imwrite(output_image_path, image)
            return output_image_path


def convert_to_pseudocode(detections):
    pseudocode = []
    i = 0
    n = len(detections)
    end_detected = False  # check if END is detected

    # decision commands
    decision_mapping = {
        "obstacle not detected": "Obstacle Not Detected",
        "line not detected": "Line Not Detected",
        "touch sensor not pressed": "Touch Sensor Not Pressed",
        "if i <= 2": "i FROM 1 TO 2",
        "if i <= 3": "i FROM 1 TO 3",
        "if i <= 5": "i FROM 1 TO 5",
        "if i <= 10": "i FROM 1 TO 10"
    }

    while i < n and not end_detected:
        element = detections[i]

        #terminator symbols
        if element['type'] == 'terminator':
            if element['command'] == 'start':
                pseudocode.append("BEGIN")
            elif element['command'] == 'end':
                pseudocode.append("END")
                end_detected = True  # Mark END

        #process symbols
        elif element['type'] == 'process':
            command = element['command'].capitalize()

            # find the next non-arrow element
            j = i + 1
            while j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                j += 1

            # if next symbol is decision with arrow connected in it of > 100 width - DO WHILE LOOP
            if j < n and detections[j]['type'] == 'decision' and \
            j + 1 < n and detections[j + 1]['type'] == 'arrow' and \
            detections[j + 1]['height'] >= 300:

                decision_command = decision_mapping.get(detections[j]['command'].lower(), "Unknown Condition")
                pseudocode.append(f"    {' '.join([word.capitalize() for word in command.split()])}")
                pseudocode.append(f"    WHILE {decision_command}")
                pseudocode.append(f"        {command}")
                pseudocode.append("    END WHILE")
                i = j  # Skip ahead to after the decision block

            #if next symbol is decision with arrow connected in it of < 100 width - WHILE LOOP
            elif j < n and detections[j]['type'] == 'decision' and \
            j + 1 < n and detections[j + 1]['type'] == 'arrow' and \
            detections[j + 1]['height'] < 300:

                #find the next non-arrow element
                pseudocode.append(f"    {' '.join([word.capitalize() for word in command.split()])}")
                j = i + 1
                while j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1

                #if next symbol is process - WHILE
                if j < n and detections[j]['type'] == 'process':
                    command = detections[j]['command'].capitalize()
                    decision_command =  decision_mapping.get(element['command'].lower(), "Unknown Condition")
                    pseudocode.append(f"    WHILE {decision_command}")
                    pseudocode.append(f"        {command}")
                    pseudocode.append("    END WHILE")
                    i = j  # Skip ahead to after the decision block

            else:
                pseudocode.append(f"    {' '.join([word.capitalize() for word in command.split()])}")


        #decision symbols (nested decision not yet implemented)
        elif element['type'] == 'decision' and \
        element['command'] in ["if i <= 2", "if i <= 3", "if i <= 5", "if i <= 10"]:

            j = i + 1
            decision_command =  decision_mapping.get(element['command'].lower(), "Unknown Condition")
            pseudocode.append(f"    FOR {decision_command}")

            #find the next non-arrow element while finding arrow of > 100 width
            while j < n and (detections[j]['type'] != 'arrow' or detections[j]['height'] <= 300):
                if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1
                elif j < n and detections[j]['type'] in ['process', 'data']:
                    command = detections[j]['command'].capitalize()
                    pseudocode.append(f"        {command}")
                    j += 1

            j += 2
            command = detections[j]['command'].capitalize()
            pseudocode.append(f"        {command}")
            pseudocode.append("    END FOR")

            i = j  # Skip to after the decision block


        elif element['type'] == 'decision' and \
        element['command'] in ["obstacle not detected", "line not detected","touch sensor not pressed"]:

            #find  next non-arrow element
            j = i + 1
            while j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                j += 1

            #if next symbol is process - WHILE
            if j < n and detections[j]['type'] == 'process':
                command = detections[j]['command'].capitalize()
                decision_command =  decision_mapping.get(element['command'].lower(), "Unknown Condition")
                pseudocode.append(f"    WHILE {decision_command}")
                pseudocode.append(f"        {command}")
                pseudocode.append("    END WHILE")
                i = j  # Skip ahead to after decision block
            else:
                pseudocode.append(f"    {' '.join([word.capitalize() for word in command.split()])}")
        i += 1

    # END will be added if not detected
    if not end_detected:
        pseudocode.append("END")

    return "\n".join(pseudocode)


def translate_pseudocode(pseudocode):
    command_mapping = {
        "Move Forward Five Seconds": "F,5",
        "Move Forward": "F",
        "Move Forward Two Seconds": "F,2",
        "Move Backward Five Seconds": "B,5",
        "Move Backward": "B",
        "Move Backward Two Seconds": "B,2",
        "Turn Left": "L",
        "Turn Left Two Times": "L,2",
        "Turn Left Five Times": "L,5",
        "Turn Right": "R",
        "Turn Right Two Times": "R,2",
        "Turn Right Five Times": "R,5",
        "Turn 180": "T,180",
        "Turn 360": "T,360",
        "Delay Two Seconds": "D,2",
        "Delay Five Seconds": "D,5",
        "Delay": "D",
        "Turn on Light": "on",
        "Turn off Light": "off",
        "Play Sound": "S",
        "Obstacle Not Detected": "obs",
        "Line Not Detected": "line",
        "Touch Sensor Not Pressed": "touch"
    }

    commands = []
    loop_stack = []

    def parse_command(line):
        line = line.strip()
        for key, value in command_mapping.items():
            if line.startswith(key):
                return f"<{value}>"
        return None

    def format_condition(condition):
        formatted_condition = condition.strip().replace(":", "").replace(" ", "_").lower()
        return command_mapping.get(condition, formatted_condition)

    for line in pseudocode.split('\n'):
        line = line.strip()

        if line.startswith("BEGIN") or line == "":
            continue  # Skip BEGIN

        elif line.startswith("FOR"):
            loop_stack.append(line)
            _, condition = line.split(' ', 1)
            #  loop count
            if "TO" in condition:
                _, to_part = condition.split("TO")
                loop_count = to_part.strip()
                commands.append(f"<for,{loop_count}>")
            else:
                commands.append("<for>")

        elif line.startswith("WHILE"):
            loop_stack.append(line)
            _, condition = line.split(' ', 1)
            formatted_condition = format_condition(condition)
            commands.append(f"<while,{formatted_condition}>")

        elif line.startswith("END FOR") or line.startswith("END WHILE"):
            if loop_stack:
                loop_stack.pop()
                commands.append("<end>")

        else:
            command = parse_command(line)
            if command:
                commands.append(command)
    return ''.join(commands)



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
        
        # Load the uploaded image
        image = cv2.imread(image_path)
        
        # Convert the image to grayscale
        warped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian adaptive thresholding
        T = threshold_local(warped, block_size=65, offset=33, method="gaussian")
        warped = (warped > T).astype("uint8") * 255
        
        # Save the processed image
        processed_image_path = "static/objects/processed_image.jpg"
        cv2.imwrite(processed_image_path, warped)
        
        # Perform detection
        detection_result = OCR_CLIENT.detect_diagram(processed_image_path)

        # Convert to Pseudocode
        pseudocode_result = convert_to_pseudocode(detection_result)

        arduino_commands = translate_pseudocode(pseudocode_result)

        # Save the image with detections
        output_image_path = OCR_CLIENT.print_result_with_ocr(detection_result, processed_image_path)

        # Upload image with detections to Firebase Storage
        blob = bucket.blob(f'detected_images/{os.path.basename(output_image_path)}')
        blob.upload_from_filename(output_image_path)
        image_url = blob.generate_signed_url(expiration=datetime.timedelta(days=7))

        # Save JSON results
        pseudocode_path = os.path.join('static/detected_images', file.filename.split('.')[0] + '.json')
        with open(pseudocode_path, 'w') as pseudocode_file:
            json.dump(pseudocode_result, pseudocode_file, indent=4)

        # Upload JSON to Firebase Storage
        pseudocode_blob = bucket.blob(f'detected_images/{os.path.basename(pseudocode_path)}')
        pseudocode_blob.upload_from_filename(pseudocode_path)
        pseudocode_url = pseudocode_blob.generate_signed_url(expiration=datetime.timedelta(days=7))

        # Save URLs to Firestore
        doc_ref = db.collection('image_data').document(file.filename.split('.')[0])
        doc_ref.set({
            'image_url': image_url,
            'pseudocode_url': pseudocode_url
        })

        # Clean up temporary files
        os.remove(image_path)
        os.remove(output_image_path)
        os.remove(pseudocode_path)

        return jsonify({
            "message": "File processed successfully",
            "image_url": image_url,
            "pseudocode_url": pseudocode_url
            "arduino_commands": arduino_commands
        })

if __name__ == '__main__':
    app.run(debug=True)
