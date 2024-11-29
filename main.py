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
import re


# Ensure the 
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
    "turn right",
]

start_end = ["start", "end"]

'''
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
'''

predefined_conditions = ["for i in range", "while obstacle not detected", "if obstacle cm ahead"]

input_output = ["read distance", "check obstacle", "set speed to slow", "set speed to medium", "set speed to high"]

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
        return "unrecognized text"  # Return invalid if symbol_type is unrecognized

    # Iterate through the relevant predefined strings
    for predefined in predefined_list:
        ratio = fuzz.ratio(predefined, normalized_text)
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = predefined

    if best_match == "for i in range" and highest_ratio >= 50:
        temp = re.findall(r'\d+', normalized_text)
        num = ''.join(temp)
        return f"i in range 1 to {num}" if 0 < len(num) < 3 else f"unknown condition ({text})"

    elif best_match == "if obstacle cm ahead" and highest_ratio >= 50:
        temp = re.findall(r'\d+', normalized_text)
        num = ''.join(temp)
        try:
            num_int = int(num)
            if num_int < 250:
                return f"obstacle {num}cm ahead"
            else:
                return f"unknown condition (distance value too high)"
        except ValueError:
            return f"obstacle 100cm ahead"

    elif best_match == "while obstacle not detected":
        return best_match if highest_ratio >= 50 else f"unknown condition ({text})"

    else:
        return best_match if highest_ratio >= 50 else f"unknown command ({text})"
    
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
                        (filtered_results[i]['command'].startswith("i in range") or filtered_results[i]['command'].startswith("while")):

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
    result = model.predict(thresh_img_3channel, conf=0.3, iou=0.7)[0]


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
            pos = y1 + 11 #add allowance just in case

        elif class_name == 'arrow':
            pos = y2 - 15 #add allowance just in case

        elif class_name == 'arrowhead':
            pos = y2 - 3 #add allowance just in case
            
        elif class_name.lower().replace("rotation", "") == 'terminator' and matched_command == 'start':
            pos = y1 - 10 #add allowance just in case

        elif class_name.lower().replace("rotation", "") == 'terminator' and matched_command == 'end':
            pos = y2 + 10 #add allowance just in case

        else:
            pos = y2

        detection_with_ocr = {
            'type': class_name.lower().replace("rotation", ""),
            'coordinates': (x, y),
            'height': height,
            'width': width,
            'command': matched_command if text != "no text detected" else text,
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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.7)
    
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
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{detection['order']}. {detection['type']}"
            if detection['command']:
                label += f" ({detection['command']})"

            # Calculate font scale based on image dimensions
                            # Calculate font scale based on image dimensions
            font_scale = ((image_width*1.25+image_height*0.75)/2)/(50/base_scale)
            

            # Draw text on the image
            if detection['type'] == "arrowhead":
                cv2.putText(image, label, (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
            elif detection['type'] == "terminator" and detection['command'] == "end":
                cv2.putText(image, label, (x1 - 25, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
            elif detection['type'] == "arrow":
                cv2.putText(image, label, (x1 , y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
            elif detection['type'] == "decision":
                cv2.putText(image, label, (x1 - 60, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
            else:
                cv2.putText(image, label, (x1 - 20, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)

        cv2.imwrite(image_path, image)
        return image_path


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
        "while obstacle not detected": "OBSTACLE NOT DETECTED",
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

            # DO WHILE LOOP
            if j < n and detections[j]['type'] == 'decision' and \
            detections[j]['command'].startswith("while") and \
            detections[j + 1]['elbow_top_left'] == True:

                decision_command = decision_mapping.get(detections[j]['command'].lower())


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


            # WHILE LOOP
            elif j < n and detections[j]['type'] == 'decision' and \
            detections[j]['command'].startswith("while") and \
            detections[j + 1]['elbow_top_left'] == False:
                pseudocode.append(f"    {command}")

                j += 1
                decision_command = decision_mapping.get(detections[j-1]['command'].lower())
                pseudocode.append(f"    WHILE {decision_command}")

                # Find the next non-arrow element while finding arrow of > 100 width
                while j < n and detections[j]['elbow_top_left'] != True and detections[j]['elbow_bottom_curved'] != True and (time.time() - start_time) < max_time:

                    if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1
                    elif j < n and detections[j]['type'] in ['process', 'data']:
                        command = capitalize_words(detections[j]['command'])
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
            detections[j]['command'].startswith("obstacle"):
                pseudocode.append(f"    {command}")

                decision_command = detections[j]['command'].upper()
                pseudocode.append(f"    IF {decision_command}")
                j += 2


                # Find the next non-arrow element while finding arrow of > 100 width
                while j < n and detections[j]['elbow_bottom_left'] != True and detections[j]['elbow_bottom_curved'] != True and (time.time() - start_time) < max_time:
                    if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1
                    elif j < n and detections[j]['type'] in ['process', 'data']:
                        command = capitalize_words(detections[j]['command'])
                        pseudocode.append(f"        {command}")
                        j += 1

                pseudocode.append("    END IF")

                while j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1

                if j < n and detections[j]['type'] in ['process', 'data']:
                    command = capitalize_words(detections[j]['command'])
                    pseudocode.append(f"    {command}")
                i = j  # Skip to after the decision block



            # FOR LOOP
            elif j < n and detections[j]['type'] == 'decision' and \
            detections[j]['command'].startswith("i in range") and \
            detections[j + 1]['elbow_top_left'] == False:
                pseudocode.append(f"    {command}")

                j += 1
                decision_command =detections[j-1]['command'].upper()
                pseudocode.append(f"    FOR {decision_command}")

                # Find the next non-arrow element while finding arrow of > 100 width
                while j < n and detections[j]['elbow_top_left'] != True and detections[j]['elbow_bottom_curved'] != True and (time.time() - start_time) < max_time:

                    if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1
                    elif j < n and detections[j]['type'] in ['process', 'data']:
                        command = capitalize_words(detections[j]['command'])
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

            #unknown text
            elif j < n and detections[j]['type'] == 'decision' and \
            (detections[j]['command'].startswith("unknown") or detections[j]['command'].startswith("no text")):
                pseudocode.append(f"    {command}")
                decision_command = capitalize_words(detections[j]['command'])
                pseudocode.append(f"    {decision_command}")
                i = j  # Skip to after the decision block

            else:
                pseudocode.append(f"    {command}")


        # Decision symbols (nested decision not yet implemented)
        elif element['type'] == 'decision' and \
        element['command'].startswith("i in range"):
        # FOR LOOP
            j = i + 1
            decision_command = element['command'].upper()
            pseudocode.append(f"    FOR {decision_command}")

            # Find the next non-arrow element while finding arrow of > 100 width
            while j < n and detections[j]['elbow_top_left'] != True and detections[j]['elbow_bottom_curved'] != True and (time.time() - start_time) < max_time:

                if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1
                elif j < n and detections[j]['type'] in ['process', 'data']:
                    command = capitalize_words(detections[j]['command'])
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
            pseudocode.append(f"    WHILE {decision_command}")

            # Find the next non-arrow element while finding arrow of > 100 width
            while j < n and detections[j]['elbow_top_left'] != True and detections[j]['elbow_bottom_curved'] != True and (time.time() - start_time) < max_time:

                if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1
                elif j < n and detections[j]['type'] in ['process', 'data']:
                    command = capitalize_words(detections[j]['command'])
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
        element['command'].startswith("obstacle"):

            decision_command = element['command'].upper()
            pseudocode.append(f"    IF {decision_command}")
            j = i + 2

            # Find the next non-arrow element while finding arrow of > 100 width
            while j < n and detections[j]['elbow_bottom_left'] != True and detections[j]['elbow_bottom_curved'] != True and (time.time() - start_time) < max_time:
                if j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                    j += 1
                elif j < n and detections[j]['type'] in ['process', 'data']:
                    command = capitalize_words(detections[j]['command'])
                    pseudocode.append(f"        {command}")
                    j += 1

            pseudocode.append("    END IF")
            while j < n and detections[j]['type'] in ['arrow', 'arrowhead']:
                j += 1

            if j < n and detections[j]['type'] in ['process', 'data']:
                command = capitalize_words(detections[j]['command'])
                pseudocode.append(f"    {command}")


            i = j  # Skip to after the decision block

        elif element['type'] == 'decision' and \
        (element['command'].startswith("unknown") or element['command'].startswith("no text")):
            decision_command = capitalize_words(element['command'])
            pseudocode.append(f"    {decision_command}")

        i += 1

    # END will be added if not detected
    if not end_detected:
        pseudocode.append("END")

    return "\n".join(pseudocode)


def translate_pseudocode(pseudocode):
    command_mapping = {
        "Move Forward": "F",
        "Move Backward": "B",
        "Turn Left": "L",
        "Turn Right": "R",
        "Set Speed To Slow": "S",
        "Set Speed To Medium": "M",
        "Set Speed To High": "H"
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
        
    def parse_if(line):
        line = line.strip()
        match = re.search(r"IF OBSTACLE (\d+)CM AHEAD", line, re.IGNORECASE) 
        if match:
            number = match.group(1)  # Extract the number
            return f"<if,{number}>" 
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
            command = parse_if(line)
            if command:
                commands.append(command)

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
    num_decision = 0
    num_symbols = 0
    command_none_count = 0
    invalid_decision_count = 0
    elbow_arrow_count = 0
    terminator_commands = []  # Store commands of terminator symbols
    errors = []  # To accumulate error messages

    # Analyze sorted_result
    for detection in sorted_result:
        label = detection['type']
        command = detection.get('command', None)

        if label not in ['arrow', 'arrowhead']:
            num_symbols += 1

            if label in ['process', 'data']:
                num_process_data += 1
                if command is None or command.startswith("unknown"):
                    command_none_count += 1

            elif label == 'decision':
                num_decision += 1
                if command is None or command.startswith("unknown"):
                    invalid_decision_count += 1

            elif label == 'terminator':
                num_terminators += 1
                if command is None or command.startswith("unknown"):
                    command_none_count += 1
                else:
                    terminator_commands.append(command.strip().lower())

        elif label == 'arrow':
            num_arrows += 1
            # Check if the arrow is an elbow arrow
            if any(
                detection.get(key, False)
                for key in ['elbow_top_left', 'elbow_bottom_curved', 'elbow_top_right', 'elbow_bottom_left']
            ):
                elbow_arrow_count += 1

        elif label == 'arrowhead':
            num_arrowheads += 1

    # Error Conditions
    if num_terminators != 2 or set(terminator_commands) != {'start', 'end'}:
        errors.append("Flowchart must contain exactly two terminators: 'start' and 'end'.")

    if abs(num_arrows - num_symbols) > 3:
        errors.append("Missing arrows.")

    if num_process_data == 0:
        errors.append("Flowchart must include at least one process or data symbol.")

    if invalid_decision_count > 0:
        errors.append("Decision symbol/s contain invalid text.")

    if num_decision > elbow_arrow_count:
        errors.append("Arrows in the decision loop are incomplete.")

    if command_none_count > 2:
        errors.append("Too many invalid texts were found in process/data symbols.")

    if abs(num_arrows - num_arrowheads) > 3:
        errors.append("Too many arrows with missing arrowheads.")

    # Minor Issues
    minor_issues = []
    if command_none_count == 1 or command_none_count == 2:
        minor_issues.append("Invalid text/s were found in process / data symbol. Replace it with a correct command syntax.")
    if 1 <= abs(num_arrows - num_symbols) <= 2:
        minor_issues.append(f"There are {num_symbols} symbols found in the flowchart but {num_arrows} arrows. Verify the pseudocode.")
    if 1 <= abs(num_arrows - num_arrowheads) <= 2:
        minor_issues.append(f"There are {num_arrows} arrows found in the flowchart but {num_arrowheads} arrowheads. Verify the pseudocode.")

    # Final Decision
    if not errors and not minor_issues:
        return {
            "status": "success",
            "error_list": "",
            "dialog_message": "No errors were found! You may click 'Next' to proceed."
        }
    elif not errors and minor_issues:
        return {
            "status": "success",
            "error_list": "\n".join(minor_issues),
            "dialog_message": "Click next to proceed. Double-check the pseudocode before running the commands on the robot!"
        }
    else:
        return {
            "status": "failed",
            "error_list": f"Following mistakes below were detected in the flowchart:\n\n" + "\n".join(errors),
            "dialog_message": "Uh oh! I think there's something wrong. Tap the error icon on the bottom for more details."
        }


def validate_pseudocode(pseudocode: str):
    # Valid commands
    valid_commands = {
        "set speed to slow",
        "set speed to medium",
        "set speed to high",
        "while obstacle not detected",
        "if obstacle (n)cm ahead",
        "move forward",
        "move backward",
        "turn left",
        "turn right",
        "start",
        "end",
        "end for",
        "end if",
        "end while",
    }

    pseudocode_lines = pseudocode.strip().split("\n")
    for_stack = []
    while_stack = []
    conditional_stack = []
    has_commands = False

    # Error message format
    def generate_error(line_no, line_text, description):
        return {
            "status": "fail",
            "line_with_error": line_no,
            "error_message": f"""
line {line_no}
    {line_text}
        ^
SyntaxError: {description}"""
        }

    # Check nested loop
    def check_nested(line_no, current_structure):
        if for_stack or while_stack or conditional_stack:
            return generate_error(
                line_no,
                pseudocode_lines[line_no - 1],
                f"nested '{current_structure}' is not allowed"
            )
        return None

    # Command checker
    def check_line(line, line_no):
        nonlocal has_commands
        line = line.strip().lower()  # Case-insensitive

        if line in {"begin", "end"}:
            return None  # BEGIN and END are checked later

        if line.startswith("for i in range 1 to "):
            try:
                n = int(line.split("to ")[1])
                if not (1 <= n <= 99):
                    raise ValueError("'n' is out of range (1-99)")
                error = check_nested(line_no, "for")
                if error:
                    return error
                for_stack.append(line_no)
            except ValueError:
                return generate_error(line_no, line, "malformed 'for i in range 1 to n' or 'n' is out of range (1-99)")
            has_commands = True

        elif line.startswith("if obstacle ") and line.endswith("cm ahead"):
            try:
                n = int(line.split("obstacle ")[1].split("cm ahead")[0])
                if n > 250:
                    raise ValueError("'n' exceeds 250")
                error = check_nested(line_no, "if")
                if error:
                    return error
                conditional_stack.append(line_no)
            except ValueError:
                return generate_error(line_no, line, "malformed 'if obstacle (n)cm ahead' or 'n' exceeds 250")
            has_commands = True

        elif line == "end for":
            if not for_stack:
                return generate_error(line_no, line, "'end for' without matching 'for i in range 1 to n'")
            for_stack.pop()

        elif line == "end if":
            if not conditional_stack:
                return generate_error(line_no, line, "'end if' without matching 'if obstacle (n)cm ahead'")
            conditional_stack.pop()

        elif line == "while obstacle not detected":
            error = check_nested(line_no, "while")
            if error:
                return error
            while_stack.append(line_no)
            has_commands = True

        elif line == "end while":
            if not while_stack:
                return generate_error(line_no, line, "'end while' without matching 'while obstacle not detected'")
            while_stack.pop()

        elif line in valid_commands:
            has_commands = True

        else:
            return generate_error(line_no, line, "unrecognized command")
        return None

    # First and last lines must be BEGIN and END
    if pseudocode_lines[0].strip().lower() != "begin":
        return generate_error(1, pseudocode_lines[0], "first line must be 'begin'")
    if pseudocode_lines[-1].strip().lower() != "end":
        return generate_error(len(pseudocode_lines), pseudocode_lines[-1], "last line must be 'end'")

    # Check lines one by one
    for line_no, line in enumerate(pseudocode_lines, start=1):
        error = check_line(line, line_no)
        if error:
            return error

    # Additional checking
    if len(pseudocode_lines) == 2 and pseudocode_lines[0].strip().lower() == "begin" and pseudocode_lines[1].strip().lower() == "end":
        return generate_error(1, pseudocode_lines[0], "'begin' and 'end' only, no commands between")
    if for_stack:
        return generate_error(for_stack[-1], pseudocode_lines[for_stack[-1] - 1], "'for i in range 1 to n' without matching 'end for'")
    if conditional_stack:
        return generate_error(conditional_stack[-1], pseudocode_lines[conditional_stack[-1] - 1], "'if obstacle (n)cm ahead' without matching 'end if'")
    if while_stack:
        return generate_error(while_stack[-1], pseudocode_lines[while_stack[-1] - 1], "'while obstacle not detected' without matching 'end while'")

    # If no errors
    return {"status": "success", "error_message": "Pseudocode is valid"}
    
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
    
    checking_result = is_valid_flowchart(sorted_result)
    
    # Checking Flowchart
    if checking_result["status"] == "failed":
             
        # Save the image with detections
        print_result(detection_result, resized_image_path)
            
        # Upload image with detections to Firebase Storage
        blob = bucket.blob(f'detected_images/{os.path.basename(resized_image_path)}')
        blob.upload_from_filename(resized_image_path)
        image_url = blob.generate_signed_url(expiration=datetime.timedelta(days=7))
    

        # Clean up temporary files
        os.remove(image_path)
        os.remove(resized_image_path)
        
        return JSONResponse({
            "status": "Failed",
            "image_url": image_url,
            "message": checking_result["dialog_message"],
            "error_list": checking_result["error_list"]
        })
        
    else:

        # Save the image with detections
        print_result_with_ocr(result, resized_image_path)
        
        # Convert to pseudocode
        pseudocode_result = convert_to_pseudocode(sorted_result)
        
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
            "message": checking_result["dialog_message"],
            "error_list": checking_result["error_list"]
        })
        

@app.post("/translate_pseudocode_from_file")
async def translate_pseudocode_from_file(file: UploadFile):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .txt file.")
    
    # Read the contents of the text file
    contents = await file.read()
    pseudocode = contents.decode('utf-8')  # Ensure it is decoded to a string

    # Validate pseudocode
    validation_result = validate_pseudocode(pseudocode)
    if validation_result["status"] == "fail":
        return {
            "status": "Failed",
            "message": "Uh oh! I think there's something wrong. Tap the error icon on the bottom for more details.",
            "error_list": validation_result["error_message"],
            "line_with_error": validation_result["line_with_error"],
            "arduino_commands": ""  # Return an empty string if validation fails
        }

    try:
        # Translate the pseudocode only if validation is successful
        arduino_commands = translate_pseudocode(pseudocode)
        return {
            "status": "Success",
            "message": "Success! Your robot is ready to go!",
            "error_list": "",
            "line_with_error": "",
            "arduino_commands": arduino_commands
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    app.run(debug=True)
