from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import json
import tempfile
import cv2
import requests
import io
import torch
from google.cloud import vision
from google.oauth2 import service_account
from google.cloud.vision_v1 import types
from PIL import Image, ImageDraw, ImageFont
import difflib
from difflib import get_close_matches
from difflib import SequenceMatcher as SM
from skimage.filters import threshold_local
import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage
import skimage.filters as filters
import numpy as np

# Ensure the necessary directories exist
for directory in ['static/objects', 'static/detected_images']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create JSON files from environment variables
google_credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
firebase_credentials_json = os.environ.get("FIREBASE_APPLICATION_CREDENTIALS_JSON")

if google_credentials_json:
    with open("/app/google-vision-config.json", "w") as google_file:
        json.dump(json.loads(google_credentials_json), google_file)

if firebase_credentials_json:
    with open("/app/firebase-config.json", "w") as firebase_file:
        json.dump(json.loads(firebase_credentials_json), firebase_file)

# Set environment variable for Google Cloud Vision API
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/google-vision-config.json"

app = FastAPI()

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/models", StaticFiles(directory="models"), name="models")
templates = Jinja2Templates(directory="templates")

# Initialize Firebase Admin
cred = credentials.Certificate("/app/firebase-config.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'psykitz-891d8.appspot.com'})

db = firestore.client()
bucket = storage.bucket()
predefined_commands = [
    "move forward",
    "move forward two times",
    "move forward five times",
    "move backward",
    "move backward two times",
    "move backward five times",
    "turn left",
    "turn right",
    "turn 180",
    "spin",
    "delay one second",
    "delay two seconds",
    "delay five seconds",
    "drive forward",
    "drive backward",
    "stop",
    "turn on led"
]

start_end = ["start", "end"]

input_output = ["check obstacle", "display distance", "set speed to slow", "set speed to medium", "set speed to fast"]

predefined_conditions = [
    "while obstacle not detected", "while line not detected", "if line detected",
    "for i in range (2)", "for i in range (3)",
    "for i in range (4)", "for i in range (5)",
    "for i in range (6)", "for i in range (7)",
    "for i in range (8)", "for i in range (9)",
]

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')
model.conf = 0.45  # confidence threshold (0-1)
model.iou = 0.7
def preprocess_image(image_path):
    # Convert the image to grayscale
    image = cv2.imread(image_path)

    warped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian adaptive thresholding
    T = threshold_local(warped, block_size=45, offset=30, method="gaussian")
    warped = (warped > T).astype("uint8") * 255

    contours, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = warped[y:y+h, x:x+w]
    


    return cropped_image


def detect_handwriting(data):
    client = vision.ImageAnnotatorClient()
    with open(data, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    response = client.document_text_detection(image=image)
    texts = response.text_annotations
    if texts:
        return texts[0].description
    else:
        return "no text detected"
        

def detect_diagram(image_path):
    # Load image
    image = cv2.imread(image_path)

    
    results = model(image)
    
    detection_result_objects = results.pandas().xyxy[0]  # Use xywh format (center x, center y, width, height)

    detection_result = []
    boxes = []
    confidences = []
    arrow_data = []
    
    for idx, prediction in detection_result_objects.iterrows():
        x_min = int(prediction['xmin'])
        y_min = int(prediction['ymin'])
        x_max = int(prediction['xmax'])
        y_max = int(prediction['ymax'])
        confidence = float(prediction['confidence'])
        class_name = prediction['name']

        # Calculate width and height
        width = x_max - x_min
        height = y_max - y_min

        # Calculate center x, y
        x = x_min + (width / 2)
        y = y_min + (height / 2)


        x1 = int(x - width // 2)
        y1 = int(y - height // 2)
        x2 = int(x + width // 2)
        y2 = int(y + height // 2)

        
        # Store arrow and arrowhead data 
        if class_name.lower() in ['arrow', 'arrowhead']:
            arrow_data.append({
                'type': class_name.lower(),
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'height': height,
                'width': width,
                'center_y': y,  # Center y of the arrow
                'center_x': x,  # Center x of the arrow
                'confidence': confidence
            })

        roi = image[y1:y2, x1:x2]
        
        roi_filename = f'cropped_image_{idx}.jpg'
        roi_path = os.path.join('static/objects', roi_filename)
        cv2.imwrite(roi_path, roi)
        text = detect_handwriting(roi_path)

        matched_command = None
        if class_name.lower() not in ['arrow', 'arrowhead']:
            matched_command = match_text_with_commands(text, class_name.lower().replace("rotation", ""))
            
        # Store bounding boxes and confidences before applying NMS
        boxes.append([x1, y1, width, height])
        confidences.append(confidence)

        if class_name.lower().replace("rotation", "") == 'decision':
            pos = y1 + 10

        elif class_name == 'arrow':
            pos = y2 - 15

        elif class_name == 'arrowhead':
            pos = y2

        elif class_name.lower().replace("rotation", "") == 'terminator' and matched_command == 'end':
            pos = y2 + 10

        else:
            pos = y2


        detection_with_ocr = {
            'type': class_name.lower().replace("rotation", ""),
            'coordinates': (x, y),
            'height': height,
            'width': width,
            'command': matched_command if text != "no text detected" else "",
            'pos': pos,
            'elbow_top_left': False,  # Default to False
            'elbow_bottom_curved': False,
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
                    # Check if arrowhead overlaps with the arrow and is in the bot half
                    if (
                        arrowhead['x2'] >= arrow['x1']
                        and arrow['center_x'] >= arrowhead['x2']
                        and arrow['y2'] >= arrowhead['y1'] >= arrow['center_y']
                        and arrow['height'] >= 45
                    ):
                        # Set elbow_top_left = True
                        for detection in detection_result:
                            if (detection['type'] == 'arrow' and
                               detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):
                              detection['elbow_bottom_curved'] = True
                              detection['pos'] -= 100
                    if (
                        arrow['x2'] >= arrowhead['x1']
                        and arrowhead['x1'] >= arrow['center_x']
                        and arrow['y2'] >= arrowhead['y1'] >= arrow['center_y']
                        and arrow['height'] >= 45
                    ):
                        # Set elbow_top_left = True
                        for detection in detection_result:
                            if (detection['type'] == 'arrow' and
                               detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):
                              detection['elbow_bottom_curved'] = True
                              detection['pos'] -= 100
    # Apply NMS 
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.45, nms_threshold=0.7)

    # Make sure indices are correct
    if len(indices) > 0:
        indices = indices.flatten()
        filtered_results = [detection_result[i] for i in indices]
    else:
        filtered_results = detection_result
     
    # Sort results by assigned position
    filtered_results.sort(key=lambda x: x["pos"])
    
    for i in range(len(filtered_results) - 1):

        if filtered_results[i]['type'] == 'arrow' and filtered_results[i - 1]['type'] == 'arrowhead' and \
                    filtered_results[i + 1]['type'] != 'arrowhead':
                        filtered_results[i], filtered_results[i - 1] = filtered_results[i - 1], filtered_results[i]

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
                    (filtered_results[i]['command'].startswith("for") or filtered_results[i]['command'].startswith("while")):
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


def match_text_with_commands(text, symbol_type=None):
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

    # Determine if the match is appropriate based on symbol type
    if symbol_type == "process" and best_match not in predefined_commands:
        return "invalid text"
    if symbol_type == "terminator" and best_match not in start_end:
        return "invalid text"
    if symbol_type == "decision" and best_match not in predefined_conditions:
        return "invalid text"
    if symbol_type == "data" and best_match not in input_output:
        return "invalid text"
    # Return the best match if the ratio is above a certain threshold, else None
    return best_match if highest_ratio >= 0.55 else "invalid text"



def print_result(detection_result, image_path):
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        # Base scale for text
        base_scale = 0.0006  # Experiment with this value as needed

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
        "for i in range (6)": "I IN RANGE 1 TO 6",
        "for i in range (7)": "I IN RANGE 1 TO 7",
        "for i in range (8)": "I IN RANGE 1 TO 8",
        "for i in range (9)": "I IN RANGE 1 TO 9",
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
            detections[j]['command'].startswith("while") and \
            detections[j + 1]['elbow_top_left'] == True:
                decision_command = decision_mapping.get(detections[j]['command'].lower(), "Unknown Condition")
                if command != "invalid text":
                    pseudocode.append(f"    {command}")    

                k = j - 1
                if decision_command != "invalid text":
                    pseudocode.append(f"    WHILE {decision_command}")

                while k < n and \
                detections[k]['coordinates'][1] - detections[k]['height'] // 2 >= \
                detections[j + 1]['coordinates'][1] - detections[j + 1]['height'] // 2:

                    k -= 1
                if detections[k]['command'] != "invalid text":
                    pseudocode.append(f"        {capitalize_words(detections[k]['command'])}")

                while k < n and detections[k]['type'] != 'decision':
                    k += 1
                    if detections[k]['type'] == 'process' or detections[k]['type'] == 'data':
                        if detections[k]['command'] != "invalid text":
                            pseudocode.append(f"        {capitalize_words(detections[k]['command'])}")
                if command != "invalid text":
                    pseudocode.append("    END WHILE")
                i = j  # Skip ahead to after the decision block


            # If the next symbol is a decision with an arrow connected and height < 300 - WHILE LOOP
            elif j < n and detections[j]['type'] == 'decision' and \
            detections[j]['command'].startswith("while") and \
            detections[j + 1]['elbow_top_left'] == False:
                if command != "invalid text":
                    pseudocode.append(f"    {command}")

                j += 1
                decision_command = decision_mapping.get(detections[j-1]['command'], "Unknown Condition")
                if decision_command != "invalid text":
                    pseudocode.append(f"    WHILE {decision_command}")

                # Find the next non-arrow element while finding arrow of > 100 width
                while j < n and detections[j]['elbow_top_left'] != True:
                    if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1
                    elif j < n and detections[j]['type'] in ['process', 'data']:
                        command = capitalize_words(detections[j]['command'])
                        if command != "invalid text":
                            pseudocode.append(f"        {command}")
                        j += 1

                j += 2
                command = capitalize_words(detections[j]['command'])
                if command != "invalid text":
                    pseudocode.append(f"        {command}")
                    pseudocode.append("    END WHILE")

                i = j  # Skip to after the decision block

            # If the next symbol is a decision with an arrow connected and height < 300 - WHILE LOOP
            elif j < n and detections[j]['type'] == 'decision' and \
            detections[j]['command'].startswith("for") and \
            detections[j + 1]['elbow_top_left'] == False:
                if command != "invalid text":
                    pseudocode.append(f"    {command}")

                j += 1
                decision_command = decision_mapping.get(detections[j-1]['command'], "Unknown Condition")
                if decision_command != "invalid text":
                    pseudocode.append(f"    FOR {decision_command}")

                # Find the next non-arrow element while finding arrow of > 100 width
                while j < n and detections[j]['elbow_top_left'] != True:
                    if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1
                    elif j < n and detections[j]['type'] in ['process', 'data']:
                        command = capitalize_words(detections[j]['command'])
                        if command != "invalid text":
                            pseudocode.append(f"        {command}")
                        j += 1

                j += 2
                command = capitalize_words(detections[j]['command'])
                if command != "invalid text":
                    pseudocode.append(f"        {command}")
                    pseudocode.append("    END FOR")

                i = j  # Skip to after the decision block
            else:
                if command != "invalid text":
                    pseudocode.append(f"    {command}")


        # Decision symbols (nested decision not yet implemented)
        elif element['type'] == 'decision' and \
        element['command'].startswith("for i in range"):
        # FOR LOOP
            j = i + 1
            decision_command = decision_mapping.get(element['command'].lower(), "Unknown Condition")
            if decision_command != "invalid text":
                pseudocode.append(f"    FOR {decision_command}")

            # Find the next non-arrow element while finding arrow of > 100 width
            while j < n and detections[j]['elbow_top_left'] != True:
                if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1
                elif j < n and detections[j]['type'] in ['process', 'data']:
                    command = capitalize_words(detections[j]['command'])
                    if command != "invalid text":
                        pseudocode.append(f"        {command}")
                    j += 1

            j += 2
            command = capitalize_words(detections[j]['command'])
            if command != "invalid text":
                pseudocode.append(f"        {command}")
                pseudocode.append("    END FOR")

            i = j  # Skip to after the decision block

        elif element['type'] == 'decision' and \
        element['command'] in ["while obstacle not detected"]:
        # WHILE LOOP
            j = i + 1
            decision_command = decision_mapping.get(element['command'].lower(), "Unknown Condition")
            if decision_command != "invalid text":
                pseudocode.append(f"    WHILE {decision_command}")

            # Find the next non-arrow element while finding arrow of > 100 width
            while j < n and detections[j]['elbow_top_left'] != True:
                if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1
                elif j < n and detections[j]['type'] in ['process', 'data']:
                    command = capitalize_words(detections[j]['command'])
                    if decision_command != "invalid text":
                        pseudocode.append(f"        {command}")
                    j += 1

            j += 2
            command = capitalize_words(detections[j]['command'])
            if decision_command != "invalid text":
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
        "Move Backward Five Times": "B,5",
        "Move Backward": "B",
        "Move Backward Two Times": "B,2",
        "Turn Left": "L",
        "Turn Right": "R",
        "Turn 180": "T,180",
        "Turn 360": "T,360",
        "Delay One Second": "D,1",
        "Delay Two Seconds": "D,2",
        "Delay Five Seconds": "D,5",
        "Drive Backward": "R",
        "Stop": "S",
        "Obstacle Not Detected": "obs",
        "Line Not Detected": "line",
        "Touch Sensor Not Pressed": "touch",
        "Turn On Led": "LED",
        "Read Distance": "DST",
        "Check Obstacle": "CHK",
        "Set Speed To Slow": "SPS",
        "Set Speed To Normal": "SPN",
        "Set Speed To Fast": "SPF"
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




@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if file.filename == '':
        return JSONResponse({
        "status": "Failed",
        "message": "No file part",
    })

    # Save the uploaded image
    image_path = os.path.join('static/objects', file.filename)
    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())

    # Preprocess
    preprocessed_img = preprocess_image(image_path)

    # Save the preprocessed image
    preprocessed_image_path = "static/objects/processed_image.jpg"
    cv2.imwrite(preprocessed_image_path, preprocessed_img)

    # Perform detection
    detection_result = detect_diagram(preprocessed_image_path)
    
            # Check if the image contains a flowchart by ensuring there are at least 3 object detections
    num_terminators = 0
    num_arrows = 0
    num_arrowheads = 0
    count_symbols = len(detection_result)


    for detection in detection_result:
        label = detection['type']
        
           
        if label == 'terminator':
            num_terminators += 1
        elif label == 'arrow':
            num_arrows += 1
        elif label == 'arrowhead':
            num_arrowheads += 1

    # Check the conditions
    if (
        count_symbols < 7 or
        num_terminators != 2 or 
        num_arrows <= 1 or 
        num_arrowheads <= 1
    ):
        # Convert 
        pseudocode_result = "Error: INVALID SYNTAX"
        arduino_commands = ""
        
        # Save the image with detections
        output_image_path = print_result(detection_result, image_path)

        # Save the pseudocode 
        pseudocode_path = os.path.join('static/detected_images', file.filename.split('.')[0] + '.txt')
        with open(pseudocode_path, 'w') as pseudocode_file:
            pseudocode_file.write(pseudocode_result)
            
        # Upload image with detections to Firebase Storage
        blob = bucket.blob(f'detected_images/{os.path.basename(output_image_path)}')
        blob.upload_from_filename(output_image_path)
        image_url = blob.generate_signed_url(expiration=datetime.timedelta(days=7))
    
        # Upload pseudocode to Firebase Storage
        pseudocode_blob = bucket.blob(f'detected_images/{os.path.basename(pseudocode_path)}')
        pseudocode_blob.upload_from_filename(pseudocode_path)
        pseudocode_url = pseudocode_blob.generate_signed_url(expiration=datetime.timedelta(days=7))

        # Clean up temporary files
        os.remove(image_path)
        os.remove(preprocessed_image_path)
        os.remove(output_image_path)
        os.remove(pseudocode_path)
        
        return JSONResponse({
            "status": "Success",
            "image_url": image_url,
            "pseudocode_url": pseudocode_url,
            "arduino_commands": arduino_commands
        })
        
    else:
        # Convert 
        pseudocode_result = convert_to_pseudocode(detection_result)
        arduino_commands = translate_pseudocode(pseudocode_result)
    
        # Save the image with detections
        output_image_path = print_result(detection_result, image_path)
        
        # Save the pseudocode 
        pseudocode_path = os.path.join('static/detected_images', file.filename.split('.')[0] + '.txt')
        with open(pseudocode_path, 'w') as pseudocode_file:
            pseudocode_file.write(pseudocode_result)    
            
        # Upload image with detections to Firebase Storage
        blob = bucket.blob(f'detected_images/{os.path.basename(output_image_path)}')
        blob.upload_from_filename(output_image_path)
        image_url = blob.generate_signed_url(expiration=datetime.timedelta(days=7))
    
        # Upload pseudocode to Firebase Storage
        pseudocode_blob = bucket.blob(f'detected_images/{os.path.basename(pseudocode_path)}')
        pseudocode_blob.upload_from_filename(pseudocode_path)
        pseudocode_url = pseudocode_blob.generate_signed_url(expiration=datetime.timedelta(days=7))


        # Clean up temporary files
        os.remove(image_path)
        os.remove(preprocessed_image_path)
        os.remove(output_image_path)
        os.remove(pseudocode_path)
    
        return JSONResponse({
            "status": "Success",
            "image_url": image_url,
            "pseudocode_url": pseudocode_url,
            "arduino_commands": arduino_commands
        })


if __name__ == '__main__':
    app.run(debug=True)
