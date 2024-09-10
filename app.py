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
from difflib import SequenceMatcher as SM
from skimage.filters import threshold_local
import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import skimage.filters as filters

# Ensure the necessary directories exist
for directory in ['static/objects', 'static/detected_images']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Set environment variables for credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/trusty-ether-434318-m3-864061dea084.json"  # For Google Cloud Vision API

app = Flask(__name__)

# Initialize Firebase Admin
cred = credentials.Certificate("/etc/secrets/psykitz-891d8-firebase-adminsdk-l7okt-38b1a73888.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'psykitz-891d8.appspot.com'})

db = firestore.client()
bucket = storage.bucket()

# Define predefined commands and symbols
predefined_commands = [
    "move forward",
    "move backward",
    "turn left",
    "turn right",
    "turn 180"
    "delay one second",
    "drive forward",
    "drive backward",
    "follow line"
]

start_end = ["start", "end"]

predefined_conditions = [
    "while obstacle not detected", "while line not detected", "if line detected",
    "for i in range (1)", "for i in range (2)", "for i in range (3)",
    "for i in range (4)", "for i in range (5)",
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
            if texts == [""]:
                return "no text detected"
            else:
                return texts[0].description
        else:
            return "no text detected"
            

    def detect_diagram(self, image_path):
        image = cv2.imread(image_path)
        custom_configuration = InferenceConfiguration(confidence_threshold=0.3, iou_threshold=0.25)
        detection_client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)
        detection_client.configure(custom_configuration)
        detection_result_objects = detection_client.infer(image, model_id=self.model_id)

        detection_result = []
        boxes = []
        confidences = []
        arrow_data = []
        
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
            
            # Store arrow and arrowhead data 
            if symbol_class.lower() in ['arrow', 'arrowhead']:
                arrow_data.append({
                    'type': symbol_class.lower(),
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'center_y': y,  # Center y of the arrow
                    'center_x': x,  # Center x of the arrow
                    'confidence': confidence
                })

            roi = image[y1:y2, x1:x2]
            roi_filename = f'cropped_image_{idx}.jpg'
            roi_path = os.path.join('static/objects', roi_filename)
            cv2.imwrite(roi_path, roi)
            text = self.perform_ocr(roi_path)

            matched_command = None
            if symbol_class.lower() not in ['arrow', 'arrowhead']:
                matched_command = self.match_text_with_commands(text)
                
            # Store bounding boxes and confidences before applying NMS
            boxes.append([x1, y1, width, height])
            confidences.append(confidence)

            if symbol_class.lower().replace("rotation", "") == 'decision':
                pos = y1 + 10

            elif symbol_class == 'arrow':
                pos = y2 - 15

            elif symbol_class == 'arrowhead':
                pos = y2

            elif symbol_class.lower().replace("rotation", "") == 'terminator' and matched_command == 'end':
                pos = y2 + 10

            else:
                pos = y2


            detection_with_ocr = {
                'type': symbol_class.lower().replace("rotation", ""),
                'coordinates': (x, y),
                'height': height,
                'width': width,
                'command': matched_command if text != "No text detected" else "",
                'pos': pos,
                'elbow_top_left': False,  # Default to False
                'orig_text': text,
                'conf': confidence

            }
            detection_result.append(detection_with_ocr)
            
        # Check for arrowhead-overlapping arrows
        for arrow in arrow_data:
            if arrow['type'] == 'arrow':
                for arrowhead in arrow_data:
                    if arrowhead['type'] == 'arrowhead':
                        # Check if arrowhead overlaps with the arrow and is in the top half
                        if (arrow['x2'] >= arrowhead['x2'] >= arrow['x1'] and
							              arrow['center_y'] >= arrowhead['y2'] >= arrow['y1']):
                            # Set elbow_top_left = True
                            for detection in detection_result:
                                if (detection['type'] == 'arrow' and
                                   detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):
                                  detection['elbow_top_left'] = True      
        # Apply NMS 
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.8)

        # Make sure indices are correct
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_results = [detection_result[i] for i in indices]
        else:
            filtered_results = detection_result
         
        # Sort results by assigned position
        filtered_results.sort(key=lambda x: x["pos"])

        for i in range(len(filtered_results) - 1):
 
            #WHILE Implementation
            if filtered_results[i]['type'] == 'decision' and filtered_results[i + 1]['type'] == 'arrowhead' and \
            filtered_results[i]['command'] in ["while obstacle not detected", "while line not detected"]:

 
                #find the next looping arrow
                j = i + 1
                n = len(filtered_results)
                while j < n and filtered_results[j]['elbow_top_left'] != True:
                    j += 1

                # Remove the next arrowhead after decision
                removed_arrowhead = filtered_results.pop(i + 1)

                # Insert it after looping arrow
                new_index = j
                if new_index < len(filtered_results):
                    filtered_results.insert(new_index, removed_arrowhead)                    

            #DO-WHILE Implementation
            if i > 0 and i + 1 < len(filtered_results) and \
                          filtered_results[i]['type'] == 'arrowhead' and \
                          filtered_results[i + 1]['type'] == 'process' and \
                          filtered_results[i - 1]['type'] == 'arrowhead':

                    removed_arrowhead = filtered_results.pop(i)

                    j = i + 1
                    while j < len(filtered_results) and filtered_results[j]['type'] != 'decision':
                        j += 1

                    new_index = j + 2
                    if new_index < len(filtered_results):
                        filtered_results.insert(new_index, removed_arrowhead)


            #FOR LOOP Implementation
            if filtered_results[i]['type'] == 'decision' and filtered_results[i + 1]['type'] == 'arrowhead' and \
            filtered_results[i]['command'] in ["for i in range (1)", "for i in range (2)", "for i in range (3)", "for i in range (4)", "for i in range (5)"]:
                #find the next arrow element with width > 100
                j = i + 1
                n = len(filtered_results)
                while j < n and filtered_results[j]['elbow_top_left'] != True:
                    j += 1

                # Remove the second arrowhead
                removed_arrowhead = filtered_results.pop(i + 1)

                # Insert it after looping arrow
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

        # Combine predefined commands and conditions
        all_predefined = predefined_commands + predefined_conditions + start_end

        # Initialize variables to track the best match and highest ratio
        best_match = None
        highest_ratio = 0

        # Iterate through all predefined strings
        for predefined in all_predefined:
            ratio = SM(None, normalized_text, predefined).ratio()

            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = predefined

        # Return the best match if the ratio is above a certain threshold, else None
        return best_match if highest_ratio >= 0.55 else "unrecognized text"

    def print_result_with_ocr(self, detection_result, image_path):
            image = cv2.imread(image_path)
            image_height, image_width = image.shape[:2]

            # Base scale for text
            base_scale = 0.0005  # Experiment with this value as needed

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
                if detection['type'] == "arrowhead":
                    cv2.putText(image, label, (x1 - 12, y1 + 30), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 255), thickness)
                elif detection['type'] == "terminator" and detection['command'] == "end":
                    cv2.putText(image, label, (x1 - 25, y2 + 10), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 255), thickness)
                elif detection['type'] == "decision":
                    cv2.putText(image, label, (x1 - 60, y1 + 10), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 255), thickness)
                else:
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
        "while obstacle not detected": "Obstacle Not Detected",
        "while line not detected": "Line Not Detected",
        "for i in range (2)": "I IN RANGE 1 TO 2",
        "for i in range (3)": "I IN RANGE 1 TO 3",
        "for i in range (4)": "I IN RANGE 1 TO 4",
        "for i in range (5)": "I IN RANGE 1 TO 5",
    }

    def capitalize_words(text):
        return ' '.join(word.capitalize() for word in text.split())

    while i < n and not end_detected:
        element = detections[i]

        # Terminator symbols
        if element['type'] == 'terminator':
            if element['command'] == 'start':
                pseudocode.append("BEGIN")
            elif element['command'] == 'end':
                pseudocode.append("END")
                end_detected = True  # Mark END

        # Process symbols
        elif element['type'] == 'process':
            command = capitalize_words(element['command'])

            # Find the next non-arrow element
            j = i + 1
            while j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                j += 1

            # If the next symbol is a decision with an arrow connected and height >= 300 - DO WHILE LOOP
            if j < n and detections[j]['type'] == 'decision' and \
            detections[j + 1]['elbow_top_left'] == True:
                decision_command = decision_mapping.get(detections[j]['command'].lower(), "Unknown Condition")
                pseudocode.append(f"    {command}")

                k = j - 1
                pseudocode.append(f"    WHILE {decision_command}")

                while k < n and \
                detections[k]['coordinates'][1] - detections[k]['height'] // 2 >= \
                detections[j + 1]['coordinates'][1] - detections[j + 1]['height'] // 2:

                    k -= 1

                pseudocode.append(f"        {capitalize_words(detections[k]['command'])}")

                while k < n and detections[k]['type'] != 'decision':
                    k += 1
                    if detections[k]['type'] == 'process' or detections[k]['type'] == 'data':
                        pseudocode.append(f"        {capitalize_words(detections[k]['command'])}")

                pseudocode.append("    END WHILE")
                i = j  # Skip ahead to after the decision block

            # If the next symbol is a decision with an arrow connected and height < 300 - WHILE LOOP
            elif j < n and detections[j]['type'] == 'decision' and \
            detections[j + 1]['elbow_top_left'] != True:

                pseudocode.append(f"    {command}")

                j = i + 1
                decision_command = decision_mapping.get(element['command'].lower(), "Unknown Condition")
                pseudocode.append(f"    WHILE {decision_command}")

                # Find the next non-arrow element while finding arrow of > 100 width
                while j < n and detections[j]['elbow_top_left'] != True:
                    if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1
                    elif j < n and detections[j]['type'] in ['process', 'data']:
                        command = capitalize_words(detections[j]['command'])
                        pseudocode.append(f"        {command}")
                        j += 1

                j += 2
                command = capitalize_words(detections[j]['command'])
                pseudocode.append(f"        {command}")
                pseudocode.append("    END WHILE")

                i = j  # Skip to after the decision block

            else:
                pseudocode.append(f"    {command}")

        # Decision symbols (nested decision not yet implemented)
        elif element['type'] == 'decision' and \
        element['command'] in ["for i in range (1)", "for i in range (2)", "for i in range (3)", "for i in range (4)", "for i in range (5)"]:
        # FOR LOOP
            j = i + 1
            decision_command = decision_mapping.get(element['command'].lower(), "Unknown Condition")
            pseudocode.append(f"    FOR {decision_command}")

            # Find the next non-arrow element while finding arrow of > 100 width
            while j < n and detections[j]['elbow_top_left'] != True:
                if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1
                elif j < n and detections[j]['type'] in ['process', 'data']:
                    command = capitalize_words(detections[j]['command'])
                    pseudocode.append(f"        {command}")
                    j += 1

            j += 2
            command = capitalize_words(detections[j]['command'])
            pseudocode.append(f"        {command}")
            pseudocode.append("    END FOR")

            i = j  # Skip to after the decision block

        elif element['type'] == 'decision' and \
        element['command'] in ["while obstacle not detected", "while line not detected"]:
        # WHILE LOOP
            j = i + 1
            decision_command = decision_mapping.get(element['command'].lower(), "Unknown Condition")
            pseudocode.append(f"    WHILE {decision_command}")

            # Find the next non-arrow element while finding arrow of > 100 width
            while j < n and detections[j]['elbow_top_left'] != True:
                if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1
                elif j < n and detections[j]['type'] in ['process', 'data']:
                    command = capitalize_words(detections[j]['command'])
                    pseudocode.append(f"        {command}")
                    j += 1

            j += 2
            command = capitalize_words(detections[j]['command'])
            pseudocode.append(f"        {command}")
            pseudocode.append("    END WHILE")

            i = j  # Skip to after the decision block

        i += 1

    # END will be added if not detected
    if not end_detected:
        pseudocode.append("END")

    return "\n".join(pseudocode)




def translate_pseudocode(pseudocode):
    command_mapping = {
        "Move Forward Five Times": "F,5",
        "Move Forward": "F",
        "Drive Forward": "DF",
        "Move Forward Two Times": "F,2",
        "Move Forward Three Times": "F,3",
        "Move Forward Four Times": "F,4",
        "Move Forward Six Times": "F,6",
        "Move Forward Seven Times": "F,7",
        "Move Forward Eight Times": "F,8",
        "Move Forward Nine Times": "F,9",
        "Move Forward Ten Times": "F,10",
        "Move Backward Five Times": "B,5",
        "Move Backward": "B",
        "Move Backward Two Times": "B,2",
        "Move Backward Three Times": "B,3",
        "Move Backward Four Times": "B,4",
        "Move Backward Six Times": "B,6",
        "Move Backward Seven Times": "B,7",
        "Move Backward Eight Times": "B,8",
        "Move Backward Nine Times": "B,9",
        "Move Backward Ten Times": "B,10",
        "Turn Left": "L",
        "Turn Left Two Times": "L,2",
        "Turn Left Three Times": "L,3",
        "Turn Left Four Times": "L,4",
        "Turn Left Five Times": "L,5",
        "Turn Left Six Times": "L,6",
        "Turn Left Seven Times": "L,7",
        "Turn Left Eight Times": "L,8",
        "Turn Left Nine Times": "L,9",
        "Turn Left Ten Times": "L,10",
        "Turn Right": "R",
        "Turn Right Two Times": "R,2",
        "Turn Right Three Times": "R,3",
        "Turn Right Four Times": "R,4",
        "Turn Right Five Times": "R,5",
        "Turn Right Six Times": "R,6",
        "Turn Right Seven Times": "R,7",
        "Turn Right Eight Times": "R,8",
        "Turn Right Nine Times": "R,9",
        "Turn Right Ten Times": "R,10",
        "Turn 180": "T,180",
        "Turn 360": "T,360",
        "Delay One Second": "D,1",
        "Delay Two Seconds": "D,2",
        "Delay Three Seconds": "D,3",
        "Delay Four Seconds": "D,4",
        "Delay Five Seconds": "D,5",
        "Delay Six Seconds": "D,6",
        "Delay Seven Seconds": "D,7",
        "Delay Eight Seconds": "D,8",
        "Delay Nine Seconds": "D,9",
        "Delay Ten Seconds": "D,10",
        "Turn on Light": "on",
        "Turn off Light": "off",
        "Play Sound": "S",
        "Reverse": "R",
        "Start": "S",
        "End": "E",
        "Obstacle Not Detected": "obs",
        "Line Not Detected": "line",
        "Touch Sensor Not Pressed": "touch"
    }

    commands = []
    loop_stack = []

    def parse_command(line):
        line = line.strip()
        # Check for exact matches
        return f"<{command_mapping.get(line, line)}>" if line in command_mapping else None

    def format_condition(condition):
        # Use the command_mapping to get the formatted condition
        formatted_condition = command_mapping.get(condition.strip(), condition.strip())
        return formatted_condition.lower().replace(" ", "_")

    for line in pseudocode.split('\n'):
        line = line.strip()

        if line.startswith("BEGIN") or line == "":
            continue  # Skip BEGIN and empty lines

        elif line.startswith("FOR"):
            loop_stack.append(line)
            _, condition = line.split(' ', 1)
            if "TO" in condition:
                _, to_part = condition.split("TO")
                loop_count = to_part.strip()
                commands.append(f"<fr,{loop_count}>")
            else:
                commands.append("<fr>")

        elif line.startswith("WHILE"):
            loop_stack.append(line)
            _, condition = line.split(' ', 1)
            formatted_condition = format_condition(condition)
            commands.append(f"<w,{formatted_condition}>")

        elif line.startswith("END FOR"):
            if loop_stack:
                loop_stack.pop()
                commands.append("<endfr>")

        elif line.startswith("END WHILE"):
            if loop_stack:
                loop_stack.pop()
                commands.append("<endw>")

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
        return jsonify({"status": "Failed",
                        "message": "No file part",
                       }), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "Failed",
                        "message": "No selected file"}), 400
    
    if file:
        # Save the uploaded image to a temporary path
        image_path = os.path.join('static/objects', file.filename)
        file.save(image_path)

        # Initialize OCR client
        OCR_CLIENT = InferenceClient(
            api_url="https://detect.roboflow.com",
            api_key="2HQ4gVVyOyZs4i2MKawd",
            model_id="flowchart-detectioo/2"
        )
        
        # Load the uploaded image
        image = cv2.imread(image_path)
        
        # Convert the image to grayscale
        warped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian adaptive thresholding
        T = threshold_local(warped, block_size=45, offset=30, method="gaussian")
        warped = (warped > T).astype("uint8") * 255
        
        contours, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = warped[y:y+h, x:x+w]
        
        # Save the preprocessed image
        preprocessed_image_path = "static/objects/processed_image.jpg"
        cv2.imwrite(preprocessed_image_path, cropped_image)
        
        # Perform detection
        detection_result = OCR_CLIENT.detect_diagram(preprocessed_image_path)

        # Check if the image contains a flowchart by ensuring there are at least 3 object detections
            # Convert to Pseudocode
        pseudocode_result = convert_to_pseudocode(detection_result)
    
        arduino_commands = translate_pseudocode(pseudocode_result)
    
            # Save the image with detections
        output_image_path = OCR_CLIENT.print_result_with_ocr(detection_result, image_path)
    
            # Upload image with detections to Firebase Storage
        blob = bucket.blob(f'detected_images/{os.path.basename(output_image_path)}')
        blob.upload_from_filename(output_image_path)
        image_url = blob.generate_signed_url(expiration=datetime.timedelta(days=7))
    
            # Save Pseudocode to Text File
        pseudocode_path = os.path.join('static/detected_images', file.filename.split('.')[0] + '.txt')
        with open(pseudocode_path, 'w') as pseudocode_file:
            pseudocode_file.write(pseudocode_result)
    
            # Upload JSON to Firebase Storage
        pseudocode_blob = bucket.blob(f'detected_images/{os.path.basename(pseudocode_path)}')
        pseudocode_blob.upload_from_filename(pseudocode_path)
        pseudocode_url = pseudocode_blob.generate_signed_url(expiration=datetime.timedelta(days=7))
    
            # Save URLs to Firestore
        doc_ref = db.collection('image_data').document(file.filename.split('.')[0])
        doc_ref.set({
            'image_url': image_url,
            'pseudocode_url': pseudocode_url,
            'arduino_commands' : arduino_commands
        })
    
            # Clean up temporary files
        os.remove(image_path)
        os.remove(preprocessed_image_path)
        os.remove(output_image_path)
        os.remove(pseudocode_path)
        
        return jsonify({
            "status": "Success",
            "image_url": image_url,
            "pseudocode_url": pseudocode_url,
            "arduino_commands": arduino_commands
        })


if __name__ == '__main__':
    app.run(debug=True)
