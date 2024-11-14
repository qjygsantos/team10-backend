from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import firebase_admin
import os
import time
import datetime
import json
import tempfile
import cv2
import requests
import io
import torch
import PIL
import difflib
import numpy as np
import pandas as pd
import ultralytics
import supervision as sv
import skimage.filters as filters
from google.cloud import vision
from google.oauth2 import service_account
from google.cloud.vision_v1 import types
from PIL import Image, ImageDraw, ImageFont, ImageOps
from difflib import get_close_matches
from difflib import SequenceMatcher as SM
from skimage.filters import threshold_otsu, threshold_local
from firebase_admin import credentials, firestore, storage
from IPython.display import Image as IPyImage
from ultralytics import YOLO
from fuzzywuzzy import fuzz

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
    "move backward",
    "turn left",
    "turn right"
]

start_end = ["start", "end"]

input_output = ["check obstacle", "set speed to slow", "set speed to medium", "set speed to fast"]

predefined_conditions = [
    "for i in range (2)", "for i in range (3)",
    "for i in range (4)", "for i in range (5)",
    "for i in range (6)", "for i in range (7)",
    "for i in range (8)", "for i in range (9)",
    "for i in range (10)", "for i in range (11)",
    "for i in range (12)", "for i in range (13)",
    "for i in range (14)", "for i in range (15)",
    "for i in range (16)", "for i in range (17)",
    "for i in range (18)", "for i in range (19)",
    "for i in range (20)",
    "while obstacle not detected", "if obstacle ahead"
]


model = YOLO('models/yolov5m-98mAP.pt')

def preprocess_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 43)

    
    return thresh

def perform_OCR(image_np):

        client = vision.ImageAnnotatorClient()
        # Convert the NumPy array to bytes
        _, encoded_image = cv2.imencode('.jpg', image_np)  # Encode as JPEG
        image_content = encoded_image.tobytes()
        image = vision.Image(content=image_content)  
        response = client.document_text_detection(image=image)
        texts = response.text_annotations
        return texts

def get_text_in_bounding_box(xmin, ymin, xmax, ymax, ocr_data):

    texts_inside_box = []

    for text_annotation in ocr_data:
        vertices = text_annotation.bounding_poly.vertices
        # Coordinates of the OCR detected text box
        text_x_min = min(vertex.x for vertex in vertices)
        text_y_min = min(vertex.y for vertex in vertices)
        text_x_max = max(vertex.x for vertex in vertices)
        text_y_max = max(vertex.y for vertex in vertices)

        # Check if text's bounding box is within the object detection box
        if (text_x_min >= xmin and text_y_min >= ymin and text_x_max <= xmax and text_y_max <= ymax):
            texts_inside_box.append(text_annotation.description)

    return ' '.join(texts_inside_box) if texts_inside_box else "no text detected"

    
def text_matching(text, symbol_type=None):
    normalized_text = text.strip().lower()

    if normalized_text == "no text detected":
        return None

    # Initialize variables to track the best match and highest ratio
    best_match = None
    highest_ratio = 0

    # Determine the relevant predefined list based on the symbol type
    if symbol_type == "process":
        predefined_list = predefined_commands
    elif symbol_type == "terminator":
        predefined_list = start_end
    elif symbol_type == "decision":
        predefined_list = predefined_conditions
    elif symbol_type == "data":
        predefined_list = input_output
    else:
        return "unrecognized text (verify manually)"  # Return invalid if symbol_type is unrecognized

    # Iterate through the relevant predefined strings
    for predefined in predefined_list:
        ratio = fuzz.WRatio(predefined, normalized_text)
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = predefined

    # Return the best match if the ratio is above a certain threshold, else invalid
    return best_match if highest_ratio >= 20 else "unrecognized text (verify manually)"
    
def check_arrows(detection_result, term_y2, arrow_data):
    for arrow in arrow_data:
        if arrow['type'] == 'arrow':
            for arrowhead in arrow_data:
                if arrowhead['type'] == 'arrowhead':

                    if (arrow['y1'] <= term_y2 and arrowhead['y1'] <= term_y2 and arrow['center_y'] >= arrowhead['y2'] >= arrow['y1']):
                        # Set elbow_top_right = True
                         for detection in detection_result:
                            if (detection['type'] == 'arrow' and
                              detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):
                              detection['elbow_top_right'] = True


                    if (
                        arrow['x2'] >= arrowhead['x1'] >= arrow['x1']
                        and arrow['y2'] >= arrowhead['y1'] >= arrow['center_y']
                        and abs(arrow['width'] - arrowhead['width']) > 30
                    ):
                        # Set elbow_bottom_curved = True
                        for detection in detection_result:
                            if (detection['type'] == 'arrow' and
                              detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):
                              detection['elbow_bottom_curved'] = True

                    if (
                        (arrow['x2'] >= arrowhead['x2'] >= arrow['x1']
                        and arrow['y2'] >= arrowhead['y1'] >= arrow['center_y']
                        and arrow['width'] >= arrowhead['width']*2) or (arrow['width'] >= arrow['height'])
                    ):
                        # Set elbow_bottom_left = True
                        for detection in detection_result:
                            if (detection['type'] == 'arrow' and
                              detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):
                              detection['elbow_bottom_left'] = True


                    elif (arrow['x2'] >= arrowhead['x2'] >= arrow['x1'] and
                                      arrow['center_y'] >= arrowhead['y2'] >= arrow['y1'] and
                                      not any(d['elbow_bottom_curved'] and d['coordinates'] == (arrow['center_x'], arrow['center_y'])
                                              for d in detection_result)):
                        # Set elbow_top_left = True
                        for detection in detection_result:
                            if (detection['type'] == 'arrow' and
                              detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):
                              detection['elbow_top_left'] = True

    return detection_result

def arrange_symbol_order(filtered_results):
    for i in range(len(filtered_results) - 1):

        if i < len(filtered_results) - 1:

            if filtered_results[i]['type'] == 'arrow' and filtered_results[i - 1]['type'] == 'arrowhead' and \
                        filtered_results[i + 1]['type'] != 'arrowhead':
                            filtered_results[i], filtered_results[i - 1] = filtered_results[i - 1], filtered_results[i]

            if filtered_results[i]['elbow_bottom_curved'] == True and filtered_results[i - 1]['type'] == 'arrow':
                            filtered_results[i], filtered_results[i - 1] = filtered_results[i - 1], filtered_results[i]

            #DO-WHILE Implementation
            if i > 0 and i + 1 < len(filtered_results) and \
                          filtered_results[i]['type'] == 'arrowhead' and \
                          filtered_results[i + 1]['type'] in ['process', 'data'] and \
                          filtered_results[i - 1]['type'] == 'arrowhead':

                    removed_arrowhead = filtered_results.pop(i)

                    j = i + 1
                    while j < len(filtered_results) and filtered_results[j]['type'] != 'decision':
                        j += 1

                    new_index = j + 2
                    if new_index < len(filtered_results):
                        filtered_results.insert(new_index, removed_arrowhead)


            #FOR and WHILE LOOP Implementation
            if filtered_results[i]['type'] == 'decision' and filtered_results[i + 1]['type'] == 'arrowhead' and \
                        (filtered_results[i]['command'].startswith("for") or filtered_results[i]['command'].startswith("while")):

                # Find the next arrow element
                j = i + 1
                n = len(filtered_results)
                while j < n and filtered_results[j]['elbow_top_left'] != True and filtered_results[j]['elbow_bottom_curved'] != True:
                    j += 1

                # Remove the second arrowhead
                removed_arrowhead = filtered_results.pop(i + 1)
                new_index = j

                if new_index < len(filtered_results):
                    filtered_results.insert(new_index, removed_arrowhead)

    return filtered_results        


def detect_diagram(thresh_image):
# Load image

    thresh_img_3channel = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels
    result = model.predict(thresh_img_3channel, conf=0.39, iou=0.78)[0]


    result_ocr = perform_OCR(thresh_image)
    
    boxes_np = result.boxes.xyxy.cpu().numpy()
    confs_np = result.boxes.conf.cpu().numpy()
    classes_np = result.boxes.cls.cpu().numpy()
    class_names_res = [result.names[int(cls)] for cls in classes_np]  # Class names

    # Create a DataFrame with the extracted data
    data = {
        'xmin': boxes_np[:, 0],
        'ymin': boxes_np[:, 1],
        'xmax': boxes_np[:, 2],
        'ymax': boxes_np[:, 3],
        'confidence': confs_np,
        'class': classes_np,
        'name': class_names_res
    }

    detection_result_objects = pd.DataFrame(data)

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

        text = get_text_in_bounding_box(x_min, y_min, x_max, y_max, result_ocr)
                

        matched_command = None
        if class_name.lower() not in ['arrow', 'arrowhead']:
            matched_command = text_matching(text, class_name.lower().replace("rotation", ""))
            
        # Store bounding boxes and confidences before applying NMS
        boxes.append([x1, y1, width, height])
        confidences.append(confidence)

        if class_name.lower().replace("rotation", "") == 'decision':
            pos = y1 + 11

        elif class_name == 'arrow':
            pos = y2 - 15

        elif class_name == 'arrowhead':
            pos = y2 - 3
            
        elif class_name.lower().replace("rotation", "") == 'terminator' and matched_command == 'start':
            pos = y1 - 10

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
            'elbow_top_right': False,
            'elbow_bottom_left' : False,
            'orig_text': text,
            'conf': confidence

        }
        detection_result.append(detection_with_ocr)
        
    return result, detection_result, boxes, confidences, arrow_data


def sort_results(detection_result, boxes, confidences, arrow_data):
    if not detection_result:
        return [] 
    # Check for arrowhead-overlapping arrows
    total_x = sum(sym['coordinates'][0] for sym in detection_result)
    avg_center_x = total_x / len(detection_result)
    term_y2 = 0
    for detection in detection_result:
        # Check if the detection is a terminator and has the command 'start'
        if detection['type'] == 'terminator' and detection['command'] == 'start':
            # Get the y2 value
            term_y2 = detection['coordinates'][1] + detection['height'] // 2

    detection_result = check_arrows(detection_result, term_y2, arrow_data)
    
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.39, nms_threshold=0.78)
    
    # Make sure indices are crrect
    if len(indices) > 0:
        indices = indices.flatten()
        filtered_results = [detection_result[i] for i in indices]
    else:
        filtered_results = detection_result
    
    # Sort results by assigned position
    filtered_results.sort(key=lambda x: x["pos"])
    
    has_elbow_top_right = any(symbol.get('elbow_top_right', False) for symbol in filtered_results)  # Use .get() to handle missing keys
    
    if has_elbow_top_right:

            # Divide symbols into list1 (left side) and list2 (right side) 
            list1 = [sym for sym in filtered_results if sym['coordinates'][0] <= avg_center_x]
            list2 = [sym for sym in filtered_results if sym['coordinates'][0] > avg_center_x]
    
            # Combine list1 and list2 to create the final sorted list order
            filtered_results = list1 + list2
    
    filtered_results = arrange_symbol_order(filtered_results)
    
    for idx, detection in enumerate(filtered_results):
        
        # Also add the order ID of symbols just in case needed
        detection["order"] = idx + 1
    
    return filtered_results

def print_result(detection_result, image_path):
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        # Base scale for text
        base_scale = 0.04  # Experiment with this value as needed

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

            elif detection["type"] == "arrowhead":
                x = x2
                y = y2

            else:
                x = detection["coordinates"][0]
                y = detection["coordinates"][1]

            x1 = int(detection["coordinates"][0] - detection["width"] // 2)
            y1 = int(detection["coordinates"][1] - detection["height"] // 2)
            x2 = int(detection["coordinates"][0] + detection["width"] // 2)
            y2 = int(detection["coordinates"][1] + detection["height"] // 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            label = f"{detection['order']}. {detection['type']}"
            if detection['command']:
                label += f" ({detection['command']})"

            # Calculate font scale based on image dimensions
                            # Calculate font scale based on image dimensions
            font_scale = ((image_width*1.25+image_height*0.75)/2)/(50/base_scale)
            

            # Draw text on the image
            if detection['type'] == "arrowhead":
                cv2.putText(image, label, (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            elif detection['type'] == "terminator" and detection['command'] == "end":
                cv2.putText(image, label, (x1 - 25, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            elif detection['type'] == "arrow":
                cv2.putText(image, label, (x1 , y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            elif detection['type'] == "decision":
                cv2.putText(image, label, (x1 - 60, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            else:
                cv2.putText(image, label, (x1 - 20, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

        output_image_path = os.path.join('static/detected_images', os.path.basename(image_path))
        cv2.imwrite(output_image_path, image)
        return output_image_path

def print_result_with_ocr(result, image_path):
            detections = sv.Detections.from_ultralytics(result)

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

            # Use the image_path variable instead of the literal string 'image_path'
            annotated_image = Image.open(image_path)
            annotated_image = box_annotator.annotate(annotated_image, detections=detections)
            annotated_image = label_annotator.annotate(annotated_image, detections=detections)

            annotated_image.save(image_path)
            return image_path


import time

def convert_to_pseudocode(detections):
    start_time = time.time()
    max_time = 3
    
    # Initialize variables
    pseudocode = []
    i = 0
    j=0
    n = len(detections)
    end_detected = False  # check if END is detected

    # decision commands
    decision_mapping = {
        "for i in range (2)": "I IN RANGE 1 TO 2",
        "for i in range (3)": "I IN RANGE 1 TO 3",
        "for i in range (4)": "I IN RANGE 1 TO 4",
        "for i in range (5)": "I IN RANGE 1 TO 5",
        "for i in range (6)": "I IN RANGE 1 TO 6",
        "for i in range (7)": "I IN RANGE 1 TO 7",
        "for i in range (8)": "I IN RANGE 1 TO 8",
        "for i in range (9)": "I IN RANGE 1 TO 9",
        "for i in range (10)": "I IN RANGE 1 TO 10",
        "for i in range (11)": "I IN RANGE 1 TO 11",
        "for i in range (12)": "I IN RANGE 1 TO 12",
        "for i in range (13)": "I IN RANGE 1 TO 13",
        "for i in range (14)": "I IN RANGE 1 TO 14",
        "for i in range (15)": "I IN RANGE 1 TO 15",
        "for i in range (16)": "I IN RANGE 1 TO 16",
        "for i in range (17)": "I IN RANGE 1 TO 17",
        "for i in range (18)": "I IN RANGE 1 TO 18",
        "for i in range (19)": "I IN RANGE 1 TO 19",
        "for i in range (20)": "I IN RANGE 1 TO 20",
        "while obstacle not detected": "OBSTACLE NOT DETECTED",
        "if obstacle ahead": "OBSTACLE AHEAD",
        "set speed to slow": "SET Speed to Slow",
        "set speed to medium": "SET Speed to Medium",
        "set speed to fast": "SET Speed to Fast"
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
        elif element['type'] in ["process", "data"]:
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
                while j < n and detections[j]['elbow_top_left'] != True and detections[j]['elbow_bottom_curved'] != True and (time.time() - start_time) < max_time:

                    if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1
                    elif j < n and detections[j]['type'] in ['process', 'data']:
                        command = capitalize_words(detections[j]['command'])
                        if command != "invalid text":
                            pseudocode.append(f"        {command}")
                        j += 1

                if detections[j]['elbow_top_left'] == True:

                    j += 2

                    while j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1

                    if j < len(detections):
                        command = capitalize_words(detections[j]['command'])
                        pseudocode.append(f"        {command}")
                        pseudocode.append("    END WHILE")

                    else:
                        pseudocode.append("    END WHILE")

                    i = j  # Skip to after the decision block

                elif detections[j]['elbow_bottom_curved'] == True:

                    j -= 1

                    while j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1

                    if j < len(detections):
                        pseudocode.append("    END WHILE")

                    else:
                        pseudocode.append("    END WHILE")

                    i = j  # Skip to after the decision block

                else:
                    pseudocode.append("    END WHILE")

                    i = j  # Skip to after the decision block

            #IF STATEMENT
            elif j < n and detections[j]['type'] == 'decision' and \
            detections[j]['command'].startswith("if"):
                if command != "invalid text":
                    pseudocode.append(f"    {command}")

                decision_command = decision_mapping.get(detections[j]['command'], "Unknown Condition")
                if decision_command != "invalid text":
                    pseudocode.append(f"    IF {decision_command}")
                j += 2


                # Find the next non-arrow element while finding arrow of > 100 width
                while j < n and detections[j]['elbow_bottom_left'] != True and detections[j]['elbow_bottom_curved'] != True and (time.time() - start_time) < max_time:
                    if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1
                    elif j < n and detections[j]['type'] in ['process', 'data']:
                        command = capitalize_words(detections[j]['command'])
                        if command != "invalid text":
                            pseudocode.append(f"        {command}")
                        j += 1

                pseudocode.append("    END IF")

                while j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1

                if j < n and detections[j]['type'] in ['process', 'data']:
                    command = capitalize_words(detections[j]['command'])
                    if command != "invalid text":
                        pseudocode.append(f"    {command}")
                i = j  # Skip to after the decision block

            # If the next symbol is a decision with an arrow connected - FOR LOOP
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
                while j < n and detections[j]['elbow_top_left'] != True and detections[j]['elbow_bottom_curved'] != True and (time.time() - start_time) < max_time:

                    if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1
                    elif j < n and detections[j]['type'] in ['process', 'data']:
                        command = capitalize_words(detections[j]['command'])
                        if command != "invalid text":
                            pseudocode.append(f"        {command}")
                        j += 1

                if detections[j]['elbow_top_left'] == True:

                    j += 2

                    while j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1

                    if j < len(detections):
                        command = capitalize_words(detections[j]['command'])
                        pseudocode.append(f"        {command}")
                        pseudocode.append("    END FOR")

                    else:
                        pseudocode.append("    END FOR")

                    i = j  # Skip to after the decision block

                elif detections[j]['elbow_bottom_curved'] == True:

                    j -= 1

                    while j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1

                    if j < len(detections):
                        pseudocode.append("    END FOR")

                    else:
                        pseudocode.append("    END FOR")

                    i = j  # Skip to after the decision block

                else:
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
            while j < n and detections[j]['elbow_top_left'] != True and detections[j]['elbow_bottom_curved'] != True and (time.time() - start_time) < max_time:

                if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1
                elif j < n and detections[j]['type'] in ['process', 'data']:
                    command = capitalize_words(detections[j]['command'])
                    if command != "invalid text":
                        pseudocode.append(f"        {command}")
                    j += 1

            if detections[j]['elbow_top_left'] == True:

                j += 2

                while j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1

                if j < len(detections):
                    command = capitalize_words(detections[j]['command'])

                    pseudocode.append(f"        {command}")
                    pseudocode.append("    END FOR")
                else:
                    pseudocode.append("    END FOR")

                i = j  # Skip to after the decision block

            elif detections[j]['elbow_bottom_curved'] == True:
                j -= 1

                while j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1

                if j < len(detections):
                    pseudocode.append("    END FOR")

                else:
                    pseudocode.append("    END FOR")

                i = j  # Skip to after the decision block

            else:

                pseudocode.append("    END FOR")

                i = j  # Skip to after the decision block

        elif element['type'] == 'decision' and \
        element['command'] in ["while obstacle not detected"]:
        # FOR LOOP
            j = i + 1
            decision_command = decision_mapping.get(element['command'].lower(), "Unknown Condition")
            if decision_command != "invalid text":
                pseudocode.append(f"    WHILE {decision_command}")

            # Find the next non-arrow element while finding arrow of > 100 width
            while j < n and detections[j]['elbow_top_left'] != True and detections[j]['elbow_bottom_curved'] != True and (time.time() - start_time) < max_time:

                if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1
                elif j < n and detections[j]['type'] in ['process', 'data']:
                    command = capitalize_words(detections[j]['command'])
                    if command != "invalid text":
                        pseudocode.append(f"        {command}")
                    j += 1

            if detections[j]['elbow_top_left'] == True:

                j += 2

                while j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1

                if j < len(detections):
                    command = capitalize_words(detections[j]['command'])

                    pseudocode.append(f"        {command}")
                    pseudocode.append("    END WHILE")
                else:
                    pseudocode.append("    END WHILE")

                i = j  # Skip to after the decision block

            elif detections[j]['elbow_bottom_curved'] == True:
                j -= 1

                while j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1

                if j < len(detections):
                    pseudocode.append("    END WHILE")

                else:
                    pseudocode.append("    END WHILE")

                i = j  # Skip to after the decision block

            else:

                pseudocode.append("    END WHILE")

                i = j  # Skip to after the decision block

        elif element['type'] == 'decision' and \
        element['command'] in ["if obstacle ahead"]:

            decision_command = decision_mapping.get(element['command'], "Unknown Condition")
            if decision_command != "invalid text":
                pseudocode.append(f"    IF {decision_command}")
            j = i + 2

            # Find the next non-arrow element while finding arrow of > 100 width
            while j < n and detections[j]['elbow_bottom_left'] != True and detections[j]['elbow_bottom_curved'] != True and (time.time() - start_time) < max_time:
                if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1
                elif j < n and detections[j]['type'] in ['process', 'data']:
                    command = capitalize_words(detections[j]['command'])
                    if command != "invalid text":
                        pseudocode.append(f"        {command}")
                    j += 1

            pseudocode.append("    END IF")
            while j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                j += 1

            if j < n and detections[j]['type'] in ['process', 'data']:
                command = capitalize_words(detections[j]['command'])
                if command != "invalid text":
                    pseudocode.append(f"    {command}")


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
        "SET Speed to Slow": "S",
        "SET Speed to Medium": "M",
        "SET Speed to Fast": "H"
    }

    commands = []
    loop_stack = []

    def parse_command(line):
        line = line.strip()
        # Perform case-insensitive lookup:
        for key, value in command_mapping.items():
            if key.lower() == line.lower():
                return f"<{value}>"  # Use original value from command_mapping
        return None

    def format_condition(condition):
        condition = condition.strip()
        # Perform case-insensitive lookup:
        for key, value in command_mapping.items():
            if key.lower() == condition.lower():
                return value.lower().replace(" ", "_") # Use original value, lowercase, and replace spaces
        return condition.lower().replace(" ", "_")

    for line in pseudocode.split('\n'):
        line = line.strip()

        if line.lower().startswith("begin") or line == "":
            continue  # Skip BEGIN and empty lines

        elif line.lower().startswith("for"):
            loop_stack.append(line)
            _, condition = line.split(' ', 1)
            if "to" in condition.lower():
                _, to_part = condition.lower().split("to")
                loop_count = to_part.strip()
                commands.append(f"<fr,{loop_count}>")
            else:
                commands.append("<fr>")

        elif line.lower().startswith("while obstacle"):
            loop_stack.append(line)
            commands.append(f"<w,obs>")


        elif line.lower().startswith("if obstacle"):
            loop_stack.append(line)
            commands.append(f"<if,obs>")

        elif line.lower().startswith("end for"):
            if loop_stack:
                loop_stack.pop()
                commands.append("<endfr>")

        elif line.lower().startswith("end while"):
            if loop_stack:
                loop_stack.pop()
                commands.append("<endw>")

        elif line.lower().startswith("end if"):
            if loop_stack:
                loop_stack.pop()
                commands.append("<endif>")

        else:
            command = parse_command(line)
            if command:
                commands.append(command)

    return ''.join(commands)


def is_valid_flowchart(sorted_result):
    
    num_terminators = 0
    num_arrows = 0
    num_arrowheads = 0
    num_process_data = 0
    num_symbols = 0
    num_decision = 0
    command_none_count = 0


    for detection in sorted_result:
        label = detection['type']
        command = detection['command']
        
        if label not in ['arrow', 'arrowhead']:
            num_symbols += 1

            if label in ['process', 'data']:
                num_process_data += 1
                if command is None:
                    command_none_count += 1
                    
            elif label == 'decision':
                num_decision += 1
                if command is None:
                    command_none_count += 1
                    
            elif label == 'terminator':
                num_terminators += 1
                if command is None:
                    command_none_count += 1
                        
        elif label == 'arrow':
            num_arrows += 1
        elif label == 'arrowhead':
            num_arrowheads += 1


    # Check the conditions
    if (
        len(sorted_result) <= 5 or 
        num_terminators <= 1 or 
        num_arrowheads <= num_arrows*0.25 or
        num_process_data == 0 or
        (num_symbols > 0 and command_none_count >= num_symbols / 2)
    ):
        return False  # Invalid flowchart

    return True  # Valid flowchart

def resize_image(image_path, base_width):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    resized_img = img.resize((base_width, hsize), PIL.Image.Resampling.LANCZOS)

    # Convert the PIL image to a NumPy array
    resized_img_np = np.array(resized_img)
    return resized_img_np

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

    image_path = os.path.join('static/objects', file.filename)
    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())

    resized_image = resize_image(image_path, 640)
    resized_image_path = "static/objects/resized_image.jpg"
    cv2.imwrite(resized_image_path, resized_image)
    
    image = cv2.imread(resized_image_path)

    # Preprocess
    preprocessed_img = preprocess_image(image)

    # Save the preprocessed image
    result, detection_result, boxes, confidences, arrow_data = detect_diagram(preprocessed_img)

    # Sort
    sorted_result = sort_results(detection_result, boxes, confidences, arrow_data)


    # Checking Flowchart
    if not is_valid_flowchart(sorted_result):
        
        pseudocode_result = "Certain symbols were not recognized properly. Please double-check your input and try again!"
        arduino_commands = ""

        # Save the image with detections
        resized_image_path = print_result_with_ocr(result, resized_image_path)

        # Save the pseudocode 
        pseudocode_path = os.path.join('static/detected_images', file.filename.split('.')[0] + '.txt')
        with open(pseudocode_path, 'w') as pseudocode_file:
            pseudocode_file.write(pseudocode_result)
            
        # Upload image with detections to Firebase Storage
        blob = bucket.blob(f'detected_images/{os.path.basename(resized_image_path)}')
        blob.upload_from_filename(resized_image_path)
        image_url = blob.generate_signed_url(expiration=datetime.timedelta(days=7))
    
        # Upload pseudocode to Firebase Storage
        pseudocode_blob = bucket.blob(f'detected_images/{os.path.basename(pseudocode_path)}')
        pseudocode_blob.upload_from_filename(pseudocode_path)
        pseudocode_url = pseudocode_blob.generate_signed_url(expiration=datetime.timedelta(days=7))

        # Clean up temporary files
        os.remove(image_path)
        os.remove(resized_image_path)
        os.remove(pseudocode_path)
        
        return JSONResponse({
            "status": "Failed",
            "message": "Certain symbols were not recognized properly. Please double-check your input and try again!",
            "image_url": image_url,
            "pseudocode_url": pseudocode_url,
            "arduino_commands": arduino_commands
        })
        
    else:
        # Convert 
        pseudocode_result = convert_to_pseudocode(sorted_result)
        arduino_commands = translate_pseudocode(pseudocode_result)
    
        # Save the image with detections
        resized_image_path = print_result_with_ocr(result, resized_image_path)
        
        # Save the pseudocode 
        pseudocode_path = os.path.join('static/detected_images', file.filename.split('.')[0] + '.txt')
        with open(pseudocode_path, 'w') as pseudocode_file:
            pseudocode_file.write(pseudocode_result)    
            
        # Upload image with detections to Firebase Storage
        blob = bucket.blob(f'detected_images/{os.path.basename(resized_image_path)}')
        blob.upload_from_filename(resized_image_path)
        image_url = blob.generate_signed_url(expiration=datetime.timedelta(days=7))
    
        # Upload pseudocode to Firebase Storage
        pseudocode_blob = bucket.blob(f'detected_images/{os.path.basename(pseudocode_path)}')
        pseudocode_blob.upload_from_filename(pseudocode_path)
        pseudocode_url = pseudocode_blob.generate_signed_url(expiration=datetime.timedelta(days=7))

        # Clean up temporary files
        os.remove(image_path)
        os.remove(pseudocode_path)
        os.remove(resized_image_path)
    
        return JSONResponse({
            "status": "Success",
            "image_url": image_url,
            "pseudocode_url": pseudocode_url,
            "arduino_commands": arduino_commands
        })
        
@app.post("/translate_pseudocode_from_file")
async def translate_pseudocode_from_file(file: UploadFile):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .txt file.")
    
    # Read the contents of the text file
    contents = await file.read()
    pseudocode = contents.decode('utf-8')  # Ensure it is decoded to a string

    try:
        # Call the function with the file content
        arduino_commands = translate_pseudocode(pseudocode)
        
        return {
            "status": "Success",
            "arduino_commands": arduino_commands
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    app.run(debug=True)
