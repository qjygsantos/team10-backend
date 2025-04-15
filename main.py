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
import gdown

# Ensure the necessary directories exist
for directory in ['static/objects', 'static/detected_images', 'static/models']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create JSON files from environment variables
google_credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON3")
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
templates = Jinja2Templates(directory="templates")

# Initialize Firebase Admin
cred = credentials.Certificate("/app/firebase-config.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'psykitz-891d8.appspot.com'})

db = firestore.client()
bucket = storage.bucket()

predefined_commands = [
    "move forward",
    "move forward seconds",
    "move backward",
    "wait seconds",
    "move backward seconds",
    "turn left", "turn right",
    "speed = low", "speed = medium", "speed = high"
]
start_end = ["start", "end"]

predefined_conditions = ["repeat times", "no obstacle detected", "obstacle detected"]

input_output = ["move forward",
    "move forward seconds",
    "move backward",
    "wait seconds",
    "move backward seconds",
    "turn left", "turn right",
    "speed = low", "speed = medium", "speed = high"]


yes_no = ["yes", "no"]
a_b_c = ["a", "b"]


# Google Drive model file ID
MODEL_FILE_ID = "1-hGAjAvmSoBaxz4vOyxccKKWwwYZVuv5"  
MODEL_PATH = "static/best.pt"  # Save model to this path

#MODEL_FILE_ID = "11CDqGVs19sf4oriLXZ6jxfGnEqe7q7DJ"  
#MODEL_PATH = "static/best.pt"  # Save model to this path

#MODEL_FILE_ID = "10e-VKPno9tlmiis10to6VqMjv6fCa2EC" 
#MODEL_PATH = "static/best.pt"  # Save model to this path

# Download the model from Google Drive to the 'models' directory
def download_model():
    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    output_path = MODEL_PATH
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# Download the model if it's not already present
download_model()

# Load the model
model = YOLO(MODEL_PATH)

def preprocess_image(image):
    #for OCR
    #grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 43)
    
    #for obj. detection
    #blurred = cv2.GaussianBlur(grey, (3, 3), 0)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    #clahe = clahe.apply(blurred)
    #thresh2 = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 43)
    #thresh2 = cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR)

    thresh = image
    thresh2 = image
    
    return thresh2, thresh
    
def resize_image(image_path, base_width):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    resized_img = img.resize((base_width, hsize), PIL.Image.Resampling.LANCZOS)

    # Convert the PIL image to a NumPy array
    resized_img_np = np.array(resized_img)
    return resized_img_np

def perform_OCR(image_np, language_hint="en-t-i0-handwrit"):

    client = vision.ImageAnnotatorClient()

    _, encoded_image = cv2.imencode('.jpg', image_np)  # Encode as JPEG
    image_content = encoded_image.tobytes()  # Convert NumPy array to bytes
    image = vision.Image(content=image_content)

    # Specify language hint
    image_context = vision.ImageContext(language_hints=[language_hint])

    response = client.document_text_detection(image=image, image_context=image_context)
    texts = response.text_annotations

    return texts

def get_text_in_bounding_box(xmin, ymin, xmax, ymax, ocr_data):
    
    texts_inside_box = []

    for text_annotation in ocr_data:
        vertices = text_annotation.bounding_poly.vertices

        # Get OCR bounding box coordinates
        x_min = min(vertex.x for vertex in vertices)
        y_min = min(vertex.y for vertex in vertices)
        x_max = max(vertex.x for vertex in vertices)
        y_max = max(vertex.y for vertex in vertices)

        # Compute center of OCR detected text
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # Compute width and height of OCR bounding box
        text_width = x_max - x_min
        text_height = y_max - y_min

        # Compute width and height of symbol bounding box
        symbol_width = xmax - xmin
        symbol_height = ymax - ymin

        # 
        size_threshold = 1.35

        # Check if center of text inside the symbol bounding box
        center_inside = xmin <= center_x <= xmax and ymin <= center_y <= ymax

        # Check if text size is not too large)
        size_ok = text_width <= size_threshold * symbol_width and text_height <= size_threshold * symbol_height

        # Add text only if both conditions met
        if center_inside and size_ok:
            texts_inside_box.append(text_annotation.description)

    return ' '.join(texts_inside_box) if texts_inside_box else "no text detected"

    
def normalize_unicode(text):
    cyrillic_to_latin = {'А': 'A', 'а': 'a', 'В': 'B', 'в': 'b', 'С': 'C', 'с': 'c', 'Д': 'D', 'д': 'd'}
    return ''.join(cyrillic_to_latin.get(char, char) for char in text)

def text_matching(text, symbol_type=None):
    normalized_text = normalize_unicode(text.strip().lower())

    if normalized_text == "no text detected":
        return normalized_text

    # Initialize variables to track the best match and highest ratio
    predefined_list = []
    best_match = None
    highest_ratio = 0

    # Determine the relevant predefined list based on the symbol type
    if symbol_type == "process":
        predefined_list = predefined_commands
    elif symbol_type == "terminator":
        predefined_list = start_end + a_b_c
    elif symbol_type == "decision":
        predefined_list = predefined_conditions
    elif symbol_type == "data":
        predefined_list = input_output
    elif symbol_type == "arrow":
        predefined_list = yes_no
    elif symbol_type == "connector":
        predefined_list = a_b_c + start_end


    # Word-level substring matching
    for predefined in predefined_list:
        if predefined in ['move forward', 'move backward', 'a', 'b', 'c']:
            continue
        predefined_words = predefined.split()

        if all(word in normalized_text for word in predefined_words):
            best_match = predefined
            highest_ratio = 100  # Perfect match for word-level substring
            break

    if best_match is None:
        for predefined in predefined_list:
            if predefined in ['move forward','obstacle detected' 'move backward', 'a', 'b', 'c']:
              continue
            if predefined in normalized_text:
                  best_match = predefined
                  highest_ratio = 100  # Perfect match for substring
                  break

    # If no substring match is found, fall back to fuzzy matching
    if best_match is None:
        # Fuzzy matching logic
        for predefined in predefined_list:
            ratio = fuzz.ratio(predefined, normalized_text)
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = predefined

    # Handle specific conditions and thresholds
    if best_match == "repeat times" and highest_ratio >= 45:
        temp = re.findall(r'\d+', normalized_text)
        if len(temp) == 0:  # No numbers detected
            return f"unknown ({text})"
        else:
            num = ''.join(temp)
            if 0 < int(num) <= 5:
                return f"repeat {num} time" if int(num) == 1 else f"repeat {num} times"
        # Add logic for "repeat {1-5}" directly
            else:
                return f"unknown ({text})"

    elif best_match in ["no obstacle detected", "obstacle detected"]:
        return best_match if highest_ratio >= 50 else f"unknown ({text})"

    elif best_match in ['start', 'end']:
        return best_match if highest_ratio >= 55 else f"unknown ({text})"

    elif best_match in ['move forward', 'move backward']:
        temp = re.findall(r'\d+', normalized_text)
        if len(temp) == 0:  # No numbers detected
            return best_match if highest_ratio >= 55 else f"unknown ({text})"
        else:
            num = ''.join(temp)
            # Add logic for "move forward/backward {1-5} seconds" directly
            if 1 <= int(num) <= 5:
                return f"{best_match} {num} second" if int(num) == 1 else f"{best_match} {num} seconds"
            else:
                return f"unknown ({text})"

    elif best_match in ['turn left', 'turn right']:
        return best_match if highest_ratio >= 50 else f"unknown ({text})"

    elif best_match in [
        "move forward seconds",
        "move backward seconds",
        "wait seconds"
    ] and highest_ratio >= 55:
        temp = re.findall(r'\d+', normalized_text)
        if len(temp) == 0:
            return f"unknown ({text})"
        num = ''.join(temp)
        if 1 <= int(num) <= 5:
            return f"{best_match.replace('seconds', '')}{num} second" if int(num) == 1 else f"{best_match.replace('seconds', '')}{num} seconds"
        else:
            return f"unknown ({text})"

    elif best_match in a_b_c:
        return best_match if highest_ratio >= 15 else f"unknown ({text})"

    else:
        if highest_ratio >= 40 and best_match not in ["move forward seconds","move backward seconds","wait seconds"]:
            return best_match
        else:
            return f"unknown ({text})"

    
def check_arrows(detection_result, arrow_data):
    for arrow in arrow_data:
        if arrow['type'] == 'arrow':
            for arrowhead in arrow_data:
                if arrowhead['type'] == 'arrowhead':

                    # Arrow pointing down
                    if abs(arrow['width'] - arrowhead['width']) < 40 and (arrow['height'] > arrow['width']) and (arrow['y2'] > arrowhead['center_y'] > arrow['center_y']) and (arrow['x1'] < arrowhead['center_x'] < arrow['x2']):

                        for detection in detection_result:

                            if (detection['type'] == 'arrow' and
                               detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):

                               detection['straight_down'] = True

                        for detection in detection_result:

                            if (detection['type'] == 'arrowhead' and \
                              detection['coordinates'] == (arrowhead['center_x'], arrowhead['center_y'])):

                              detection['head_straight_down'] = True

                    # Arrow pointing up
                    if abs(arrow['width'] - arrowhead['width']) < 40 and (arrow['height'] > arrow['width']) and (arrow['y1'] < arrowhead['center_y'] < arrow['center_y']) and (arrow['x1']< arrowhead['center_x'] < arrow['x2']):

                        for detection in detection_result:
                            if (detection['type'] == 'arrow' and
                               detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):

                               detection['straight_up'] = True

                    # Elbow Arrow pointing down
                    if (arrow['x2'] >= arrowhead['x1'] >= arrow['x1'] and \
                        arrow['y2'] >= arrowhead['y1'] >= arrow['center_y'] and \
                        abs(arrow['width'] - arrowhead['width']) > 40):

                        for detection in detection_result:

                            if (detection['type'] == 'arrow' and \
                              detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):

                              detection['elbow_bottom_curved'] = True

                        for detection in detection_result:

                            if (detection['type'] == 'arrowhead' and \
                              detection['coordinates'] == (arrowhead['center_x'], arrowhead['center_y'])):

                              detection['head_elbow_bottom_curved'] = True

                    # Also Elbow arrow pointing down
                    if ((arrow['x2'] >= arrowhead['x2'] >= arrow['x1'] and \
                         arrow['y2'] >= arrowhead['y1'] >= arrow['center_y'] and \
                         ((arrow['width'] >= arrowhead['width']*2) or (arrow['width'] >= arrow['height'])))):

                        for detection in detection_result:

                            if (detection['type'] == 'arrow' and \
                              detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):

                              detection['elbow_bottom_left'] = True

                        for detection in detection_result:

                            if (detection['type'] == 'arrowhead' and \
                              detection['coordinates'] == (arrowhead['center_x'], arrowhead['center_y'])):

                              detection['head_elbow_bottom_left'] = True

                    # Arrow pointing left and right
                    if abs(arrow['height'] - arrowhead['height']) < 40 and \
                     arrow['width'] > arrow['height'] and \
                      (arrow['y2'] > arrowhead['center_y'] > arrow['y1']) and \
                       (arrow['x1']< arrowhead['center_x'] < arrow['x2']) and \
                        (not any((d['elbow_bottom_left'] and d['coordinates'] == (arrow['center_x'], arrow['center_y'])) or (d['elbow_bottom_curved'] and d['coordinates'] == (arrow['center_x'], arrow['center_y'])) for d in detection_result)):

                        for detection in detection_result:

                            if (detection['type'] == 'arrow' and \
                              detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):

                              detection['straight_leftRight'] = True

                    # Arrow pointing left
                    if abs(arrow['height'] - arrowhead['height']) < 40 and \
                     (arrow['width'] > arrow['height']) and \
                      (arrow['y2'] > arrowhead['center_y'] > arrow['y1']) and \
                       (arrow['x1']< arrowhead['center_x'] < arrow['center_x']) and \
                        (not any((d['elbow_bottom_left'] and d['coordinates'] == (arrow['center_x'], arrow['center_y'])) or (d['elbow_bottom_curved'] and d['coordinates'] == (arrow['center_x'], arrow['center_y'])) for d in detection_result)):

                        for detection in detection_result:

                            if (detection['type'] == 'arrow' and \
                              detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):

                              detection['straight_left'] = True

                    # Arrow pointing right
                    if abs(arrow['height'] - arrowhead['height']) < 40 and \
                     (arrow['width'] > arrow['height']) and \
                      (arrow['y2'] > arrowhead['center_y'] > arrow['y1']) and \
                       (arrow['center_x']< arrowhead['center_x'] < arrow['x2']) and \
                        (not any((d['elbow_bottom_left'] and d['coordinates'] == (arrow['center_x'], arrow['center_y'])) or (d['elbow_bottom_curved'] and d['coordinates'] == (arrow['center_x'], arrow['center_y'])) for d in detection_result)):

                        for detection in detection_result:

                            if (detection['type'] == 'arrow' and \
                              detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):

                              detection['straight_right'] = True

                    # Elbow upward
                    if (arrow['x2'] >= arrowhead['center_x'] >= arrow['x1'] and \
                        arrow['center_y'] >= arrowhead['y2'] >= arrow['y1'] and \
                        arrow['height'] >= arrow['width']):

                        for detection in detection_result:

                            if (detection['type'] == 'arrow' and
                              detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):

                              detection['elbow_upward'] = True

                    # Elbow downward
                    if (arrow['x2'] >= arrowhead['center_x'] >= arrow['x1'] and \
                        arrow['center_y'] <= arrowhead['y1'] <= arrow['y2'] and \
                        abs(arrow['width'] - arrowhead['width']) > 40):

                        for detection in detection_result:

                            if (detection['type'] == 'arrow' and
                              detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):

                              detection['elbow_downward'] = True


                    # Elbow arrow pointing up left with width greater than height
                    if (arrow['center_x'] >= arrowhead['x2'] >= arrow['x1'] and \
                        arrow['center_y'] >= arrowhead['y2'] >= arrow['y1'] and \
                        (arrow['width'] >= arrow['height'])):

                        for detection in detection_result:

                            if (detection['type'] == 'arrow' and
                              detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):

                              detection['elbow_top_left_width'] = True

                        for detection in detection_result:

                            if (detection['type'] == 'arrowhead' and
                              detection['coordinates'] == (arrowhead['center_x'], arrowhead['center_y']) and
                              not any(
                                  d['head_elbow_bottom_curved'] == True and d['coordinates'] == (arrowhead['center_x'], arrowhead['center_y']) or
                                  d['head_elbow_bottom_left'] == True and d['coordinates'] == (arrowhead['center_x'], arrowhead['center_y'])
                                  for d in detection_result
                              )):

                              detection['head_elbow_top_left_width'] = True


                    # Elbow arrow pointing up left
                    if (arrow['x2'] >= arrowhead['x2'] >= arrow['x1'] and \
                        arrow['center_y'] >= arrowhead['y2'] >= arrow['y1'] and \
                        not any((d['elbow_bottom_curved'] and d['coordinates'] == (arrow['center_x'], arrow['center_y'])) or (d['elbow_bottom_left'] and d['coordinates'] == (arrow['center_x'], arrow['center_y'])) for d in detection_result)):

                        for detection in detection_result:

                            if (detection['type'] == 'arrow' and
                              detection['coordinates'] == (arrow['center_x'], arrow['center_y'])):

                              detection['elbow_top_left'] = True

                        for detection in detection_result:

                            if (detection['type'] == 'arrowhead' and
                              detection['coordinates'] == (arrowhead['center_x'], arrowhead['center_y']) and
                              not any(
                                  d['head_elbow_bottom_curved'] == True and d['coordinates'] == (arrowhead['center_x'], arrowhead['center_y']) or
                                  d['head_elbow_bottom_left'] == True and d['coordinates'] == (arrowhead['center_x'], arrowhead['center_y'])
                                  for d in detection_result
                              )):

                              detection['head_elbow_top_left'] = True

    return detection_result

def arrange_symbol_order(filtered_results):
    start_time = time.time()
    max_time = 3
    n = len(filtered_results)

    try:
        for i in range(len(filtered_results) - 1):
            if i < len(filtered_results) - 1:
                if (filtered_results[i]['type'] == 'arrow' and
                    filtered_results[i - 1]['type'] == 'arrowhead' and
                    filtered_results[i + 1]['type'] != 'arrowhead'):
                    filtered_results[i], filtered_results[i - 1] = filtered_results[i - 1], filtered_results[i]

                if (filtered_results[i]['elbow_bottom_curved'] == True and
                    filtered_results[i - 1]['type'] == 'arrow'):
                    filtered_results[i], filtered_results[i - 1] = filtered_results[i - 1], filtered_results[i]

                # DO-WHILE Implementation
                if (i > 0 and i + 1 < len(filtered_results) and
                    filtered_results[i]['type'] == 'arrowhead' and
                    filtered_results[i + 1]['type'] in ["process", "data"] and
                    filtered_results[i - 1]['type'] == 'arrowhead'):
                    removed_arrowhead = filtered_results.pop(i)
                    j = i + 1
                    while (j + 1 < len(filtered_results) and
                          filtered_results[j]['type'] != 'decision' and
                          filtered_results[j + 1]['elbow_top_left'] != True):
                        j += 1
                    new_index = j + 2
                    if new_index < len(filtered_results):
                        filtered_results.insert(new_index, removed_arrowhead)
                        

                # FOR and WHILE LOOP Implementation
                if (filtered_results[i]['type'] == 'decision' and
                    any(filtered_results[j]['type'] == 'arrowhead' and
                        filtered_results[j]['head_elbow_top_left'] == True and
                        (abs(filtered_results[i]['x2'] - filtered_results[j]['x1']) < 50 or
                        abs(filtered_results[i]['x1'] - filtered_results[j]['x2']) < 50)
                        for j in range(max(0, i - 3), min(i + 6, len(filtered_results)))
                        if j != i)):
                    filtered_results[i]['for_while'] = True


                if filtered_results[i]['type'] == 'decision':
                    decision_x1 = filtered_results[i]['x1']
                    decision_x2 = filtered_results[i]['x2']
                    for j in range(max(0, i - 4), min(i + 1, len(filtered_results))):  # Check previous 4 and next 2 items
                        if j != i and filtered_results[j]['type'] == 'arrowhead' and (decision_x1 <= filtered_results[j]['x1'] <= decision_x2) and \
                          filtered_results[j].get('head_elbow_top_left_width', False):
                            filtered_results[i]['for_while_horizontal'] = True

                if (filtered_results[i]['type'] == 'decision' and i + 4 < n):
                    next_four_symbols = filtered_results[i + 1:i + 5]
                    num_straight_leftright = sum(1 for symbol in next_four_symbols if symbol.get('straight_leftRight', False))
                    num_elbow_top_left = sum(1 for symbol in next_four_symbols if symbol.get('elbow_top_left', False))
                    num_elbow_top_left_width = sum(1 for symbol in next_four_symbols if symbol.get('elbow_top_left_width', False))
                    num_both = num_straight_leftright + num_elbow_top_left + num_elbow_top_left_width
                    if num_both >= 2:
                        filtered_results[i]['for_while_horizontal'] = True

                if (filtered_results[i]['type'] == 'decision' and filtered_results[i]['for_while'] == True and
                    filtered_results[i + 1]['type'] == 'arrowhead' and filtered_results[i + 1]['head_elbow_top_left'] == False):
                    filtered_results[i], filtered_results[i + 1] = filtered_results[i + 1], filtered_results[i]

                if (filtered_results[i]['type'] == 'decision' and filtered_results[i + 1]['type'] == 'arrowhead' and
                    filtered_results[i + 1]['head_elbow_top_left'] == True):
                    j = i + 1
                    while (j + 1 < n and filtered_results[j]['elbow_top_left'] != True and
                          (time.time() - start_time) < max_time):
                        j += 1
                    removed_arrowhead = filtered_results.pop(i + 1)
                    new_index = j
                    if new_index < len(filtered_results):
                        filtered_results.insert(new_index, removed_arrowhead)

                # IF-ELSE
                if (filtered_results[i]['type'] == 'decision' and
                    any(filtered_results[j]['command'] == 'yes' and
                        filtered_results[j]['x1'] < filtered_results[i]['coordinates'][0]
                        for j in [i + 1, i + 2, i + 3] if j < len(filtered_results))):
                    filtered_results[i]['reverse_decision'] = True

    except (IndexError, KeyError, TypeError) as e:
        # Log the error (if logging is enabled) and return the current results
        print(f"Error encountered: {e}")
        return filtered_results

    return filtered_results


def detect_diagram(thresh2, thresh):

    result_ocr = perform_OCR(thresh)
    result = model.predict(thresh2, conf=0.3, iou=0.7)[0]

    
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
        if class_name.lower() not in ['arrowhead']:
            matched_command = text_matching(text, class_name.lower().replace("rotation", ""))
            
        # Store bounding boxes and confidences before applying NMS
        boxes.append([x1, y1, width, height])
        confidences.append(confidence)

        if class_name.lower().replace("rotation", "") == 'decision':
            pos = y1 + 10

        elif class_name == 'arrow':
            pos = y2 - 10

        elif class_name == 'arrowhead':
            pos = y2

        elif class_name.lower().replace("rotation", "") == 'terminator' and matched_command == 'end':
            pos = y2 + 10

        else:
            pos = y2
            
        if class_name.lower().replace("rotation", "") == 'data':
            if matched_command.lower().startswith(("move forward", "move backward", "turn left", "turn right", "wait")):
                    class_name = 'process'
                
        if class_name.lower().replace("rotation", "") == 'connector' and matched_command in ['start', 'end']:
            class_name = 'terminator'
            
        if class_name.lower().replace("rotation", "") == 'terminator' and matched_command in ['a', 'b', 'c']:
            class_name = 'connector'
                
        detection_with_ocr = {
            'type': class_name.lower().replace("rotation", ""),
            'coordinates': (x, y),
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'height': height,
            'width': width,
            'command': matched_command if text != "no text detected" else text,
            'pos': pos,
            'orig_text': text,
            'conf': confidence,
            'reverse_decision': False,
            'for_while': False,
            'for_while_horizontal': False,
            'straight_leftRight': False,
            'straight_left': False,
            'straight_right': False,
            'straight_up': False,
            'straight_down': False,
            'elbow_top_left': False,
            'elbow_top_left_width': False,
            'elbow_bottom_left': False,
            'elbow_bottom_curved': False,
            'elbow_upward': False,
            'elbow_downward': False,
            'head_straight_down': False,
            'head_elbow_top_left': False,
            'head_elbow_top_left_width': False,
            'head_elbow_bottom_left': False,
            'head_elbow_bottom_curved': False
        }
        detection_result.append(detection_with_ocr)

    return result, detection_result, boxes, confidences, arrow_data


def sort_results(detection_result, boxes, confidences, arrow_data):

    def sort_symbols_in_place(filtered_results):
        target_classes = ['process', 'data', 'terminator', 'connector']
        n = len(filtered_results)

        for i in range(n):
            # Ensure the current symbol is in target_classes
            if filtered_results[i]['type'] not in target_classes:
                continue

            # Start with the current symbol
            current = filtered_results[i]
            current_pos = i
            farthest_valid_pos = None

            # Compare the current symbol with all subsequent symbols
            for j in range(i + 1, n):
                if filtered_results[j]['type'] in target_classes:
                    next_symbol = filtered_results[j]

                    # Check the overlap condition on the y-axis and the x-axis condition
                    if (
                        not (current['y2'] < next_symbol['y1'] or current['y1'] > next_symbol['y2']) and
                        next_symbol['x1'] <= next_symbol['x2'] <= current['x1']
                    ):
                        farthest_valid_pos = j  # Update the farthest valid position

            # If a valid farthest position is found, swap the symbols
            if farthest_valid_pos is not None:
                filtered_results[current_pos], filtered_results[farthest_valid_pos] = (
                    filtered_results[farthest_valid_pos],
                    filtered_results[current_pos],
                )

        return filtered_results


    if not detection_result:
        return detection_result

    # Check for arrowhead-overlapping arrows
    total_x = sum(sym['coordinates'][0] for sym in detection_result)
    avg_center_x = total_x / len(detection_result)
    to_remove = [] #for connectors

    detection_result = check_arrows(detection_result, arrow_data)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.35, nms_threshold=0.65)

    if len(indices) > 0:
        indices = indices.flatten()
        filtered_results = [detection_result[i] for i in indices]
    else:
        filtered_results = detection_result

    filtered_results.sort(key=lambda x: x["pos"])


    # Filter out connectors that are inside other symbols
    for i, symbol in enumerate(filtered_results):
        if symbol['type'] == 'connector':

            center_x = (symbol['x1'] + symbol['x2']) / 2
            center_y = (symbol['y1'] + symbol['y2']) / 2

            for j, other_symbol in enumerate(filtered_results):
                if i != j and other_symbol['type'] not in ['arrow', 'arrowhead']:  # Avoid comparing to itself or arrowheads

                    x1_o, y1_o, x2_o, y2_o = other_symbol['x1'], other_symbol['y1'], other_symbol['x2'], other_symbol['y2']

                    if (x1_o <= center_x <= x2_o and
                        y1_o <= center_y <= y2_o):
                        to_remove.append(i)
                        break  # No need to check further for this connector

    filtered_results = [filtered_results[k] for k in range(len(filtered_results)) if k not in to_remove]

    # 1. Check for connectors and dynamically divide into columns
    connectors = [sym for sym in filtered_results if sym['type'] == 'connector']
    num_connectors = len(connectors)

    if connectors and num_connectors % 2 == 0:  # Ensure even number of connectors
        # Sort connectors by x-coordinate
        connectors.sort(key=lambda x: x['coordinates'][0])

        # Calculate column boundaries
        column_bounds = []
        for i in range(0, num_connectors, 2):
            avg_x = (connectors[i]['coordinates'][0] + connectors[i + 1]['coordinates'][0]) / 2
            column_bounds.append(avg_x)

        # Determine the column for each symbol
        def assign_to_column(symbol, bounds):
            x = symbol['coordinates'][0]
            for i, bound in enumerate(bounds):
                if x < bound:
                    return i  # Assign to column i (leftmost)
            return len(bounds)  # Assign to the last column (rightmost)

        # Assign symbols to respective columns
        columns = [[] for _ in range(len(column_bounds) + 1)]
        for sym in filtered_results:
            col_index = assign_to_column(sym, column_bounds)
            columns[col_index].append(sym)

        # Sort each column by y-coordinate
        for col in columns:
            col.sort(key=lambda x: x['pos'])
            sort_symbols_in_place(col)

        # Flatten columns back into a single list
        filtered_results = [sym for col in columns for sym in col]

        filtered_results = arrange_symbol_order(filtered_results)

        # Assign order to symbols
        for idx, detection in enumerate(filtered_results):
            detection["order"] = idx + 1

        return filtered_results

    else:
        # 2. Arrange and finalize symbol order
        filtered_results = sort_symbols_in_place(filtered_results)
        filtered_results = arrange_symbol_order(filtered_results)

        # Assign order to symbols
        for idx, detection in enumerate(filtered_results):
            detection["order"] = idx + 1

        return filtered_results


def print_result(detection_result, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image.shape[:2]

    # Base scale for text
    base_scale = 0.02

    for detection in detection_result:

        x1 = int(detection["coordinates"][0] - detection["width"] // 2)
        y1 = int(detection["coordinates"][1] - detection["height"] // 2)
        x2 = int(detection["coordinates"][0] + detection["width"] // 2)
        y2 = int(detection["coordinates"][1] + detection["height"] // 2)

        if detection["type"] == "arrow":
            x = x2
            y = detection["coordinates"][1]

        else:
            x = detection["coordinates"][0]
            y = detection["coordinates"][1]


        # Get color
        conf = detection.get("conf", 0) * 100  # Convert confidence to percentage

        if conf < 50:
            color = (0, 0, 255)  # Red
        elif conf >= 50:
            color = (0, 255, 0)  # Green
        else:
            color = (255, 255, 255)  # Default to white (if confidence is below 25)


        if detection["type"] not in ["arrowhead"]:

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Format the label and command
            label = f"symbol: {detection['type']}"
            command_text = f"text: \"{detection['command']}\""

            # Calculate font scale
            font_scale = ((image_width * 1.25 + image_height * 0.75) / 2) / (50 / base_scale)

            # Calculate text positions
            text_x, text_y = x1 - 20, y1 + 5

            if detection['type'] == "connector":
                text_x, text_y = x2, y1
            elif detection['type'] == "terminator" and detection['command'] == "end":
                text_x, text_y = x1 - 25, y2 + 10
            elif detection['type'] == "arrow":
                text_x, text_y = x2, y1 + 30
            elif detection['type'] == "decision":
                text_x, text_y = x2,y1

            # Calculate text sizes
            text_size_label = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_size_command = cv2.getTextSize(command_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]

            # Set background size and spacing between lines
            if detection['type'] == "arrow" and (detection['command'] == "no text detected" or detection['command'].startswith('unknown')):
                bg_width = text_size_label[0] + 5
            else:
                bg_width = max(text_size_label[0], text_size_command[0]) + 17

            line_spacing = int(text_size_label[1] * 2)

            if detection['type'] == "connector" or ((detection['type'] == "arrow" and ((detection['command'] == "no text detected") or (detection['command'].startswith('unknown'))))):
                bg_height = text_size_label[1] + line_spacing
            else:
                bg_height = text_size_label[1] + text_size_command[1] + line_spacing + 3

            # Draw white background for the text

            cv2.rectangle(image,
                          (text_x, text_y - bg_height),
                          (text_x + bg_width, text_y),
                          (255, 255, 0),
                          -1)

            # Render the label
            cv2.putText(image, label, (text_x + 5, text_y - bg_height + text_size_label[1] + 5),
                        cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 0), 2)

            # Render the command text below the label
            if (detection['type'] != "connector") and (detection['type'] != "arrow" and detection['command'] != "no text detected"):
                
                cv2.putText(image, command_text, (text_x + 5, text_y - bg_height + text_size_label[1] + line_spacing),
                            cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 0), 2)

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


def convert_to_pseudocode(detections):
    start_time = time.time()
    max_time = 3
    # Initialize variables
    pseudocode = []
    i = 0
    j= 0
    k = 0
    l = 0
    n = len(detections)
    end_detected = False  # check if END is detected


    def capitalize_words(text):
        return ' '.join(word.capitalize() for word in text.split())

    def sort_symbols_in_place(filtered_results):
        target_classes = ['process', 'data', 'terminator', 'connector']
        n = len(filtered_results)

        for i in range(n):
            # Ensure the current symbol is in target_classes
            if filtered_results[i]['type'] not in target_classes:
                continue

            # Start with the current symbol
            current = filtered_results[i]
            current_pos = i
            farthest_valid_pos = None

            # Compare the current symbol with all subsequent symbols
            for j in range(i + 1, n):
                if filtered_results[j]['type'] in target_classes:
                    next_symbol = filtered_results[j]

                    # Check the overlap condition on the y-axis and the x-axis condition
                    if (
                        (not ((current['y2'] < next_symbol['y1']) or (current['y1'] > next_symbol['y2']))) and
                        (next_symbol['x1'] <= next_symbol['x2'] <= current['x2'])
                    ):
                        farthest_valid_pos = j  # Update the farthest valid position

            # If a valid farthest position is found, swap the symbols
            if farthest_valid_pos is not None:
                filtered_results[current_pos], filtered_results[farthest_valid_pos] = (
                    filtered_results[farthest_valid_pos],
                    filtered_results[current_pos],
                )

        return filtered_results

    while i < len(detections):
        try:
            element = detections[i]

            # Terminator symbols
            if i < len(detections) and element['type'] == 'terminator':
                if element['command'] == 'start':
                        pseudocode.append("start")
                elif element['command'] == 'end':
                        pseudocode.append("end")
                        end_detected = True  # Mark END

            # Process symbols
            elif i < len(detections) and element['type'] in ["process", "data"]:
                command = element['command']

                pseudocode.append(f"    {command}")

            # Decision symbols (nested decision not yet implemented)

            #HORIZONTAL REPEAT/WHILE LOOP
            elif i < len(detections) and element['type'] == 'decision' and \
            element['for_while_horizontal'] == True:
                commands = []
                j = i + 1
                decision_command = element['command']
                decision_x1 = element['x1']
                decision_x2 = element['x2']
                decision_y2 = element['y2']

                if not decision_command.startswith("repeat"):
                    pseudocode.append(f"    while {decision_command}")
                else:
                    pseudocode.append(f"    {decision_command}")

                while j < len(detections) and not (((decision_x1 <= detections[j]['coordinates'][0] <= decision_x2) and (decision_y2 < detections[j]['coordinates'][1]))) and (time.time() - start_time) < max_time:

                    if j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                        commands.append(detections[j])
                        j += 1

                    elif j < len(detections) and (detections[j]['type'] in ['process', 'data', 'terminator']):
                        command = detections[j]['command']
                        commands.append(detections[j])
                        j += 1

                commands.sort(key=lambda x: x['pos'])
                commands = sort_symbols_in_place(commands)

                for command in commands:
                    if command['type'] in ['process', 'data', 'terminator']:
                        pseudocode.append(f"        {command['command']}")

                if decision_command.startswith("repeat"):
                    pseudocode.append("    endrepeat")
                else:
                    pseudocode.append("    endwhile")

                i = j


            # DO WHILE LOOP / DO REPEAT LOOP
            elif i < len(detections) and element['type'] == 'decision' and \
            element['for_while_horizontal'] == False and \
            detections[i + 1]['elbow_top_left'] == True:

                do_while_y_coord = detections[i]['coordinates'][1]

                decision_command = detections[i]['command']

                if decision_command.startswith("repeat"):
                    pseudocode.append(f"    {decision_command}") #now append the decision symbol command
                else:
                    pseudocode.append(f"    while {decision_command}") #check if repeat or while loop

                # go upwards til it finds the end of loop body
                k = i - 1
                while k < len(detections) and \
                detections[k]['coordinates'][1] - detections[k]['height'] // 2 >= \
                detections[i + 1]['coordinates'][1] - detections[i + 1]['height'] // 2:

                    k -= 1

                if detections[k]['type'] not in ["arrow", "arrowhead"]:

                    pseudocode.append(f"        {detections[k]['command']}") #append the first item of loop body

                #now go downwards to get the other items til it goes back to decision symbol
                while k < len(detections) and detections[k]['type'] != 'decision' and detections[k]['coordinates'][1] != do_while_y_coord:

                    k += 1

                    if detections[k]['type'] in ['process', 'data', 'terminator']:

                        pseudocode.append(f"        {detections[k]['command']}")

                    if k < len(detections) and detections[k]['type'] == 'decision' and \
                    detections[k]['for_while'] == False and \
                    detections[k + 1]['elbow_top_left'] == False and \
                    detections[k]['command'] in ["obstacle detected", "no obstacle detected"]:

                        falseBranch = []
                        trueBranch = []
                        decision_x = detections[k]['coordinates'][0]
                        decision_y = element['coordinates'][1]
                        decision_x1 = detections[k]['x1']
                        decision_x2 = detections[k]['x2']

                        if detections[k]['reverse_decision'] == True:
                            reverse = True
                        else:
                            reverse = False

                        decision_command = detections[k]['command']
                        pseudocode.append(f"        if {decision_command}")

                        l = k
                        l += 1

                        while l < len(detections) and detections[l]['type'] in ['arrow', 'arrowhead']:

                            if detections[l]['type'] == 'arrowhead' and \
                            (decision_x1 < detections[l]['coordinates'][0] < decision_x2) and \
                            (detections[l]['coordinates'][1] > decision_y):

                                break

                            l += 1

                        # Find the next non-arrow element
                        while l < len(detections) and (time.time() - start_time) < max_time:

                            # Check if the current detection is of type 'arrowhead' and its x-coordinate is within the decision boundaries
                            if detections[l]['type'] == 'arrowhead' and decision_x1 < detections[l]['coordinates'][0] < decision_x2:
                                break

                            elif l < len(detections) and detections[l]['type'] in ['arrow', 'arrowhead']:
                                l += 1

                            elif l < len(detections) and (detections[l]['type'] in ['process', 'data', 'terminator', 'decision']):

                                command = detections[l]['command']
                                if detections[l]['x1'] < decision_x:
                                    if reverse:
                                        trueBranch.append(command)
                                    else:
                                        falseBranch.append(command)

                                else:
                                    if reverse:
                                        falseBranch.append(command)
                                    else:
                                        trueBranch.append(command)
                                l += 1

                        if trueBranch:
                            for command in trueBranch:
                                pseudocode.append(f"           {command}")

                        else:
                            pseudocode.append(f"           do nothing")


                        if falseBranch:
                            pseudocode.append("       else")
                            for command in falseBranch:
                                pseudocode.append(f"           {command}")
                            pseudocode.append("       endif")

                        else:
                            pseudocode.append("       endif")


                        falseBranch = []
                        trueBranch = []


                        k = l  # Skip to after the decision block

                if decision_command.startswith("repeat"):
                    pseudocode.append("    endrepeat")
                else:
                    pseudocode.append("    endwhile")


                i = k  # Skip ahead to after the decision block


            #REPEAT/WHILE LOOP
            elif i < len(detections) and element['type'] == 'decision' and \
            element['for_while'] == True and \
            element['for_while_horizontal'] == False and \
            detections[i+1]['elbow_top_left'] == False:


                falseBranch = []

                decision_x = element['coordinates'][0]
                j = i + 1
                decision_command = element['command']


                if not decision_command.startswith("repeat"):
                    pseudocode.append(f"    while {decision_command}")
                else:
                    pseudocode.append(f"    {decision_command}")

                # Find the next non-arrow element while finding arrow of > 100 width
                while j < len(detections) and detections[j]['elbow_top_left'] != True and (time.time() - start_time) < max_time:

                    if j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:

                        if ( (detections[j]['x1'] < detections[j]['x2'] < decision_x) or (detections[j]['x2'] > detections[j]['x1'] > decision_x) ):
                          popped_item = detections.pop(j)
                          falseBranch.append(popped_item)
                          j -= 1 # Decrement j here
                        j += 1

                    elif j < len(detections) and (detections[j]['type'] in ['process', 'data', 'terminator', 'decision']):
                        command = detections[j]['command']
                        if ((detections[j]['x1'] < detections[j]['x2'] < decision_x) or (detections[j]['x2'] > detections[j]['x1'] > decision_x)):
                            popped_item = detections.pop(j)
                            falseBranch.append(popped_item)

                            j -= 1 # Decrement j here
                        else:
                            pseudocode.append(f"        {command}")
                        j += 1


                if j < len(detections) and detections[j]['elbow_top_left'] == True:

                    j += 1

                    if j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1

                    elif j < len(detections) and (detections[j]['type'] in ['process', 'data', 'terminator', 'decision']):
                        command = detections[j]['command']
                        if ( (detections[j]['x1'] < detections[j]['x2'] < decision_x) or (detections[j]['x2'] > detections[j]['x1'] > decision_x) ):
                            popped_item = detections.pop(j)
                            falseBranch.append(popped_item)
                            j -= 1 # Decrement j here
                        else:
                            pseudocode.append(f"        {command}")
                        j += 1

                    while j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1

                    if j < len(detections):
                        while (detections[j]['x1'] < detections[j]['x2'] < decision_x) or (detections[j]['x2'] > detections[j]['x1'] > decision_x):
                            popped_item = detections.pop(j)
                            falseBranch.append(popped_item)

                            j -= 1 # Decrement j here
                            j += 1

                        command = detections[j]['command']
                        pseudocode.append(f"        {command}")

                        if decision_command.startswith("repeat"):
                            pseudocode.append("    endrepeat")
                        else:
                            pseudocode.append("    endwhile")

                    else:
                        if decision_command.startswith("repeat"):
                            pseudocode.append("    endrepeat")
                        else:
                            pseudocode.append("    endwhile")

                    detections[j + 1:j + 1] = falseBranch


                    falseBranch = []


                    i = j  # Skip to after the decision block


                elif j < len(detections) and (detections[j]['elbow_bottom_curved'] == True or detections[j]['elbow_bottom_left'] == True):

                    j -= 1

                    while j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1

                    if j < len(detections):
                        if decision_command.startswith("repeat"):
                            pseudocode.append("    endrepeat")
                        else:
                            pseudocode.append("    endwhile")

                    else:
                        if decision_command.startswith("repeat"):
                            pseudocode.append("    endrepeat")
                        else:
                            pseudocode.append("    endwhile")

                    detections[j + 1:j + 1] = falseBranch

                    falseBranch = []

                    i = j  # Skip to after the decision block

                else:
                    if decision_command.startswith("repeat"):
                        pseudocode.append("    endrepeat")
                    else:
                        pseudocode.append("    endwhile")

                    detections[j + 1:j + 1] = falseBranch

                    falseBranch = []



                    i = j  # Skip to after the decision block


            #IF-ELSE CONDITION
            elif i < len(detections) and element['type'] == 'decision' and \
            element['for_while'] == False and \
            element['for_while_horizontal'] == False and \
            detections[i + 1]['elbow_top_left'] == False and \
            detections[i]['command'] in ["obstacle detected", "no obstacle detected"]:

                j = i + 1
                if element['reverse_decision'] == True:
                    reverse = True
                else:
                    reverse = False

                falseBranch = []
                trueBranch = []
                decision_x = element['coordinates'][0]
                decision_y = element['coordinates'][1]
                decision_x1 = element['x1']
                decision_x2 = element['x2']

                decision_command = element['command']
                pseudocode.append(f"    if {decision_command}")

                while j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                  if detections[j]['type'] == 'arrowhead' and (decision_x1 < detections[j]['coordinates'][0] < decision_x2) and (detections[j]['coordinates'][1] > decision_y):
                      break
                  j += 1

                # Find the next non-arrow element while finding arrow of > 100 width
                while j < len(detections) and (time.time() - start_time) < max_time:
                    # Check if the current detection is of type 'arrowhead' and its x-coordinate is within the decision boundaries
                    if j < len(detections) and detections[j]['type'] == 'arrowhead' and decision_x1 < detections[j]['coordinates'][0] < decision_x2:
                        break

                    elif j < len(detections) and detections[j]['type'] in ['arrow', 'arrowhead']:
                        j += 1

                    elif j < len(detections) and (detections[j]['type'] in ['process', 'data','terminator', 'decision']):
                        command = detections[j]['command']
                        if detections[j]['x1'] < decision_x:
                            if reverse:
                                trueBranch.append(command)
                            else:
                                falseBranch.append(command)

                        else:
                            if reverse:
                                falseBranch.append(command)
                            else:
                                trueBranch.append(command)
                        j += 1

                if trueBranch:
                    for command in trueBranch:
                        pseudocode.append(f"        {command}")

                else:
                    pseudocode.append(f"        do nothing")


                if falseBranch:
                    pseudocode.append("    else")
                    for command in falseBranch:
                        pseudocode.append(f"        {command}")
                    pseudocode.append("    endif")

                else:
                    pseudocode.append("    endif")


                falseBranch = []
                trueBranch = []


                i = j  # Skip to after the decision block

            #NONE OF THE ABOVE
            elif i < len(detections) and element['type'] == 'decision' and \
            (element['command'].startswith("unknown") or element['command'].startswith("no text")):
                decision_command = element['command']
                pseudocode.append(f"    {decision_command}")

            i += 1

        except KeyError as e:
            print(f"Error: Missing key {e} in detection element {detections[i]}")
            pseudocode.append("end")  # Append "stop" if there's a key error.
            break

        except IndexError as e:
            print(f"Error: Index out of range. {e}")
            pseudocode.append("end")  # Append "stop" if there's an index error.
            break
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            pseudocode.append("end")  # Append "stop" for any other unexpected errors.
            break

    # END will be added if not detected
    if not end_detected:
        pseudocode.append("end")

    return "\n".join(pseudocode)





def translate_pseudocode(pseudocode):
    command_mapping = {
        "move forward": "F",
        "move backward": "B",
        "turn left": "L",
        "turn right": "R",
        "speed = low": "SL",
        "speed = medium": "SM",
        "speed = high": "SH",
        "get distance": "GD"
    }

    commands = []
    loop_stack = []

    def parse_move(line):
        line = line.strip().lower()
        match = re.match(r"move (forward|backward)(?: (\d+) second(?:s)?)?", line)
        if match:
            direction = "F" if match.group(1) == "forward" else "B"
            duration = match.group(2)
            return f"<{direction}>" if not duration or duration == "1" else f"<{direction},{duration}>"
        return None

    def parse_wait(line):
        line = line.strip().lower()
        match = re.match(r"wait (\d+) second(?:s)?", line)
        if match:
            duration = int(match.group(1))
            if 1 <= duration <= 5:
                return f"<D,{duration}>"
        return None

    def parse_condition(line):
        if "if no obstacle detected" in line.lower():
            return "<IF,NO>"
        elif "if obstacle detected" in line.lower():
            return "<IF,O>"
        return None

    for line in pseudocode.split("\n"):
        line = line.strip()

        if line.lower().startswith("start") or line == "":
            continue  # Skip 'start' and empty lines

        elif line.lower().startswith("repeat"):
            loop_stack.append("repeat")
            match = re.match(r"repeat (\d+) time(?:s)?", line.lower())
            loop_count = match.group(1) if match else "1"
            commands.append(f"<REPEAT,{loop_count}>")

        elif line.lower().startswith("while no obstacle detected"):
            loop_stack.append("while")
            commands.append("<WHILE,NO>")

        elif line.lower().startswith("while obstacle detected"):
            loop_stack.append("while")
            commands.append("<WHILE,O>")

        elif line.lower().startswith("endrepeat"):
            if loop_stack and loop_stack[-1] == "repeat":
                loop_stack.pop()
                commands.append("<ENDREPEAT>")

        elif line.lower().startswith("endwhile"):
            if loop_stack and loop_stack[-1] == "while":
                loop_stack.pop()
                commands.append("<ENDWHILE>")

        elif line.lower().startswith("if"):
            condition = parse_condition(line)
            if condition:
                loop_stack.append("if")
                commands.append(condition)

        elif line.lower().startswith("else"):
            if loop_stack and loop_stack[-1] == "if":
                commands.append("<ELSE>")

        elif line.lower().startswith("endif"):
            if loop_stack and loop_stack[-1] == "if":
                loop_stack.pop()
                commands.append("<ENDIF>")

        else:
            move_command = parse_move(line)
            wait_command = parse_wait(line)

            if move_command:
                commands.append(move_command)
            elif wait_command:
                commands.append(wait_command)
            else:
                for key, value in command_mapping.items():
                    if key in line.lower():
                        commands.append(f"<{value}>")

    return ''.join(commands)




def is_valid_flowchart(sorted_result):
    total = len(sorted_result)
    val = 0
    num_terminators = 0
    num_arrows = 0
    num_process_data = 0
    num_decision = 0
    num_symbols = 0
    num_connectors = 0
    num_arrowheads = 0

    command_none_count = 0
    invalid_decision_count = 0

    elbow_arrow_count = 0
    upward_arrow_count = 0

    terminator_commands = []  # Store commands of terminator symbols
    low_confidence_symbols = []  # Store low confidence symbols
    unrecognized_commands = []  # Store unrecognized commands

    errors = []  # To accumulate error messages
    warnings = []  # To accumulate warning messages

    # Analyze sorted_result
    for detection in sorted_result:
        label = detection['type']
        command = detection.get('command', None)
        confidence = detection.get('conf', 100)  # Default to 100 if not present

        if label not in ['arrow', 'arrowhead']:


            if confidence < 0.50:
                low_confidence_symbols.append(f"{label} (conf: {confidence})")

            if label in ['process', 'data']:
                num_symbols += 1
                num_process_data += 1
                if command.startswith("no text") or command.startswith("unknown"):
                    command_none_count += 1
                    unrecognized_commands.append(f"{label}: {command}")

            elif label == 'connector':
                num_symbols += 0.5
                num_connectors += 1

            elif label == 'decision':
                num_symbols += 2
                num_decision += 1

                if command.startswith("no text") or command.startswith("unknown"):
                    command_none_count += 1
                    unrecognized_commands.append(f"{label}: {command}")

            elif label == 'terminator':
                num_symbols += 1
                num_terminators += 1
                if command.startswith("no text") or command.startswith("unknown"):
                    command_none_count += 1
                    unrecognized_commands.append(f"{label}: {command}")
                else:
                    terminator_commands.append(command.strip().lower())

        elif label == 'arrow':
            if confidence < 0.50:
                low_confidence_symbols.append(f"{label} (conf: {confidence})")
            num_arrows += 1
            if any(
                detection.get(key, False)
                for key in ['elbow_upward', 'elbow_downward']
            ):
                elbow_arrow_count += 1

            if any(
                detection.get(key, False)
                for key in ['straight_up']
            ):
                upward_arrow_count += 1

        elif label == 'arrowhead':
            num_arrowheads += 1

    # Error Conditions

    if num_symbols == 0:
        errors.append("No symbols found in the flowchart.")

    if num_arrows == 0:
        errors.append("No arrows found in the flowchart.")

    if num_process_data == 0:
        errors.append("Flowchart must include at least one process or data symbol.")

    if num_terminators < 2 or not all(x in terminator_commands for x in ['start', 'end']):
        errors.append("Flowchart must contain both the 'start' and 'end' terminators.")

    if num_connectors % 2 != 0:
        errors.append("Flowchart connectivity error: Missing connector link.")




    if num_symbols <= 10:
        if num_decision == 0:
            val = abs(num_symbols - num_arrowheads)

            if val > 1:
                errors.append("Flowchart connectivity error: Missing arrows")
        else:

            if num_symbols >= num_arrowheads:
                val = abs(num_symbols - num_arrowheads) / max(num_symbols, num_arrowheads)
            else:
                val = abs(num_arrowheads - num_symbols) / max(num_symbols, num_arrowheads)

            if val >= 0.4:
                errors.append("Flowchart connectivity error: Missing arrows")

    if 20 >= num_symbols > 10:
        if num_decision == 0:
            val = abs(num_symbols - num_arrowheads)

            if val > 2:
                errors.append("Flowchart connectivity error: Missing arrows")
        else:

            if num_symbols >= num_arrowheads:
                val = abs(num_symbols - num_arrowheads) / max(num_symbols, num_arrowheads)
            else:
                val = abs(num_arrowheads - num_symbols) / max(num_symbols, num_arrowheads)

            if val >= 0.28:
                errors.append("Flowchart connectivity error: Missing arrows")


    if 30 >= num_symbols > 20:
        if num_decision == 0:
            val = abs(num_symbols - num_arrowheads)

            if val > 3:
                errors.append("Flowchart connectivity error: Missing arrows")
        else:

            if num_symbols >= num_arrowheads:
                val = abs(num_symbols - num_arrowheads) / max(num_symbols, num_arrowheads)
            else:
                val = abs(num_arrowheads - num_symbols) / max(num_symbols, num_arrowheads)

            if val >= 0.23:
                errors.append("Flowchart connectivity error: Missing arrows")


    if num_symbols > 30:
        if num_decision == 0:
            val = abs(num_symbols - num_arrowheads)

            if val > 4:
                errors.append("Flowchart connectivity error: Missing arrows")
        else:

            if num_symbols >= num_arrowheads:
                val = abs(num_symbols - num_arrowheads) / max(num_symbols, num_arrowheads)
            else:
                val = abs(num_arrowheads - num_symbols) / max(num_symbols, num_arrowheads)

            if val >= 0.19:
                errors.append("Flowchart connectivity error: Missing arrows")


    if command_none_count >= 1:
        errors.append(f"One or more symbols contain unrecognized commands or no command at all: {', '.join(unrecognized_commands)}")

    # Warnings

    if low_confidence_symbols:
        warnings.append(f"At least one low confidence symbol was found ({', '.join(low_confidence_symbols)}).")

    if elbow_arrow_count != 0 and abs(num_symbols - elbow_arrow_count ) >= 3:
        warnings.append("Flowchart connectivity warning: check for missing arrows")

    # Final Decision
    if not errors and not warnings:
        return {
            "status": "success",
            "error_list": "",
            "dialog_message": "No errors were found! You may click 'Next' to proceed."
        }
    elif not errors and warnings:
        return {
            "status": "success",
            "error_list": "\n".join(warnings),
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
        "turn left",
        "turn right",
        "speed = low",
        "speed = medium",
        "speed = high",
        "get distance",
        "do nothing",
    }

    pseudocode_lines = pseudocode.strip().split("\n")
    loop_stack = []
    conditional_stack = []

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

    # Check nested structures
    def check_nested(line_no, current_structure):
        if loop_stack or conditional_stack:
            return generate_error(
                line_no,
                pseudocode_lines[line_no - 1],
                f"nested '{current_structure}' is not supported"
            )
        return None

    # Command checker
    def check_line(line, line_no):
        line = line.strip().lower()  # Case-insensitive

        if line in {"start", "end"}:
            return None  # Start and Stop are checked later

        if line.startswith("move forward") or line.startswith("move backward"):
            parts = line.split()
            if len(parts) == 2 and parts[1] == "forward" or parts[1] == "backward":
                return None  # "move forward" or "move backward" (default 1 sec)
            try:
                if len(parts) == 4 and parts[2].isdigit() and parts[3] == "seconds":
                    seconds = int(parts[2])
                    if 1 <= seconds <= 5:
                        return None  # Valid time range
                elif len(parts) == 4 and parts[2] == "1" and parts[3] == "second":
                    return None  # "move forward 1 second"
                else:
                    raise ValueError("malformed 'move forward/backward {1-5} seconds'")
            except (ValueError, IndexError):
                return generate_error(line_no, line, "malformed 'move forward/backward {1-5} seconds'")

        elif line.startswith("wait"):
            parts = line.split()
            try:
                if len(parts) == 3 and parts[1].isdigit() and parts[2] == "seconds":
                    seconds = int(parts[1])
                    if 1 <= seconds <= 5:
                        return None  # Valid time range for wait command
                    else:
                        raise ValueError("wait time out of range (1-5 seconds)")
                elif len(parts) == 3 and parts[1] == "1" and parts[2] == "second":
                    return None  # "wait 1 second"
                else:
                    raise ValueError("malformed 'wait {1-5} seconds'")
            except (ValueError, IndexError):
                return generate_error(line_no, line, "malformed 'wait {1-5} seconds'")



        elif line.startswith("repeat"):
            try:
                times = int(line.split()[1])
                if not (1 <= times <= 5):
                    raise ValueError("repeat count out of range (1-5 times)")
                error = check_nested(line_no, "repeat")
                if error:
                    return error
                loop_stack.append(line_no)
            except (ValueError, IndexError):
                return generate_error(line_no, line, "malformed 'repeat {1-5} times'")

        elif line == "endrepeat":
            if not loop_stack:
                return generate_error(line_no, line, "'endrepeat' without matching 'repeat'")
            loop_stack.pop()

        elif line.startswith("while"):
            if line not in {"while no obstacle detected", "while obstacle detected"}:
                return generate_error(line_no, line, "unrecognized 'while' condition")
            error = check_nested(line_no, "while")
            if error:
                return error
            loop_stack.append(line_no)

        elif line == "endwhile":
            if not loop_stack:
                return generate_error(line_no, line, "'endwhile' without matching 'while'")
            loop_stack.pop()

        elif line.startswith("if"):
            if line not in {"if no obstacle detected", "if obstacle detected"}:
                return generate_error(line_no, line, "unrecognized 'if' condition")
            error = check_nested(line_no, "if")
            if error:
                return error
            conditional_stack.append(line_no)

        elif line == "else":
            if not conditional_stack:
                return generate_error(line_no, line, "'else' without matching 'if'")

        elif line == "endif":
            if not conditional_stack:
                return generate_error(line_no, line, "'endif' without matching 'if'")
            conditional_stack.pop()

        elif line in valid_commands:
            return None

        else:
            return generate_error(line_no, line, "unrecognized command")

        return None

    # First and last lines must be Start and Stop
    if pseudocode_lines[0].strip().lower() != "start":
        return generate_error(1, pseudocode_lines[0], "first line must be 'start'")
    if pseudocode_lines[-1].strip().lower() != "end":
        return generate_error(len(pseudocode_lines), pseudocode_lines[-1], "last line must be 'end'")

    # Check lines one by one
    for line_no, line in enumerate(pseudocode_lines, start=1):
        error = check_line(line, line_no)
        if error:
            return error

    # Additional checks
    if len(pseudocode_lines) == 2 and pseudocode_lines[0].strip().lower() == "start" and pseudocode_lines[1].strip().lower() == "end":
        return generate_error(1, pseudocode_lines[0], "'start' and 'end' only, no commands between")

    if loop_stack:
        return generate_error(loop_stack[-1], pseudocode_lines[loop_stack[-1] - 1], "unclosed 'repeat' or 'while'")

    if conditional_stack:
        return generate_error(conditional_stack[-1], pseudocode_lines[conditional_stack[-1] - 1], "unclosed 'if'")

    # If no errors
    return {"status": "success", "error_message": "Pseudocode is valid"}



    

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

    resized_image = resize_image(image_path, 1280)
    resized_image_path = "static/objects/resized_image.jpg"
    cv2.imwrite(resized_image_path, resized_image)
    
    image = cv2.imread(resized_image_path)

    # Preprocess
    preprocessed_img, preprocessed_ocr = preprocess_image(image)

    # Detect the preprocessed image
    result, detection_result, boxes, confidences, arrow_data = detect_diagram(preprocessed_img, preprocessed_ocr)
    
    # Extra Sorting
    sorted_result = sort_results(detection_result, boxes, confidences, arrow_data)

    # Checking Flowchart
    checking_result = is_valid_flowchart(sorted_result)
    
    if checking_result["status"] == "failed":
             
        # Save the image with detections
        print_result(sorted_result, resized_image_path)
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
        os.remove(resized_image_path)
        
        return JSONResponse({
            "status": "Failed",
            "image_url": image_url,
            "pseudocode_url": pseudocode_url,
            "message": checking_result["dialog_message"],
            "error_list": checking_result["error_list"],
            "arduino_commands": ""
        })
        
    else:

        # Save the image with detections
        print_result(sorted_result, resized_image_path)
        
        # Convert to Pseudo and String
        pseudocode_result = convert_to_pseudocode(sorted_result)
        arduino_commands = translate_pseudocode(pseudocode_result)
        
        command_list = re.findall(r'<[^>]*>', arduino_commands)
        arduino_commands_text = "\n".join(command_list)

        # Save the pseudocode 
        pseudocode_path = os.path.join('static/detected_images', file.filename.split('.')[0] + '.txt')
        with open(pseudocode_path, 'w') as pseudocode_file:
            pseudocode_file.write(pseudocode_result)    

        # Save the Arduino Command
        arduino_command_path = os.path.join('static/detected_images', file.filename.split('.')[0] + '_serial.txt')
        with open(arduino_command_path, 'w') as arduino_file:
            arduino_file.write(arduino_commands_text)    
            
        # Upload image with detections to Firebase Storage
        blob = bucket.blob(f'detected_images/{os.path.basename(resized_image_path)}')
        blob.upload_from_filename(resized_image_path)
        image_url = blob.generate_signed_url(expiration=datetime.timedelta(days=7))
    
        # Upload pseudocode to Firebase Storage
        pseudocode_blob = bucket.blob(f'detected_images/{os.path.basename(pseudocode_path)}')
        pseudocode_blob.upload_from_filename(pseudocode_path)
        pseudocode_url = pseudocode_blob.generate_signed_url(expiration=datetime.timedelta(days=7))

        # Upload arduino commands to Firebase Storage
        arduino_blob = bucket.blob(f'detected_images/{os.path.basename(arduino_command_path)}')
        arduino_blob.upload_from_filename(arduino_command_path)
        arduino_url = arduino_blob.generate_signed_url(expiration=datetime.timedelta(days=7))
        
        # Clean up temporary files
        os.remove(image_path)
        os.remove(pseudocode_path)
        os.remove(arduino_command_path)
        os.remove(resized_image_path)

        print(pseudocode_url)
        print(arduino_commands)
        print(arduino_url)
    
        return JSONResponse({
            "status": "Success",
            "image_url": image_url,
            "pseudocode_url": pseudocode_url,
            "message": checking_result["dialog_message"],
            "error_list": checking_result["error_list"],
            "arduino_commands": arduino_commands,
            "arduino_url": arduino_url
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
