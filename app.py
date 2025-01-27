from fake_api import *
import supervision as sv
from flask import Flask, request, jsonify, session, render_template, make_response
from flask_session import Session
from flask_cors import CORS
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from openai import AzureOpenAI
import json
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import base64
from io import BytesIO
from agent_workflow import *


prev_object_class = None
prev_image_data = None
prev_masks = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'  # Use filesystem-based sessions
Session(app)

# GroundingDINO+SAM configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GROUNDING_DINO_CONFIG_PATH = r"C:\Users\admin\Downloads\ground\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = r"C:\Users\admin\Downloads\groundingdino_swint_ogc.pth"
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = r"C:\Users\admin\Downloads\sam_vit_h_4b8939.pth"
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.35
NMS_THRESHOLD = 0.82

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def object_segment(image, CLASSES):
    if "the" not in CLASSES:
        CLASSES = "the " + CLASSES
    CLASSES = [CLASSES]

    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _ 
        in detections]
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    print(f"After NMS: {len(detections.xyxy)} boxes")
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    cv2.imwrite("dino_image.jpg", annotated_frame)
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    cv2.imwrite("sam_image.jpg", annotated_image)
    return detections.mask


def process_symbol(s):
    lines = s.split('\n')
    if lines and lines[0] == "```html" and lines[-1] == "```":
        return '\n'.join(lines[1:-1])  # 删除第一行和最后一行
    return s


def process_message(message):
    result=api_answer(message)
    return result


@app.before_request
def initialize_session():
    if 'option_messages' not in session:
        session['option_messages'] = []  
    # if 'ui_messages' not in session:
    #     session['ui_messages'] = []  
        
    session['ui_messages'] = []  
    session['data_agent_messages'] = []  # Initialize an empty list for messages
    session['data_agent_replacements_json']= []  # Initialize an empty list for messages
    # session['data_agent_messages'].append(message_template('system',system_prompt))# 
@app.route('/')
def index():
    session.clear()  # 清除会话数据
    return render_template('index.html')


@app.route('/option', methods=['POST'])
def option(user_message):
    system_prompt = """
        **Instructions**:
        You are an advanced image editing assistant capable of handling a variety of image editing requests. 
        The allowable operations are defined in the following list: ["recolor", "brightness", "contrast", "saturation"]. 
        Your task is to interpret the user's request and select the appropriate operation type from this predefined list, 
        then provide corresponding options.

        **Guidelines**:
        1. **Identify the Request Type:** Based on the user's request, determine which operation type from the predefined list best matches the user's need.
        2. **Object Identification:** If applicable (mainly for recolor operations), identify the object within the image that the user wants to modify.
        3. **Provide Adjustment Options:**
            - For **recolor** operations, follow previous instructions for identifying objects and provide a gradient of at least 8 hexadecimal color codes that represent variations from light to dark shades of that color
            - For adjustments like **brightness**, **contrast**, or **saturation**, provide adjustment values as floating-point numbers where 1 represents the original effect. Values less than 1 indicate a reduction in intensity, while values greater than 1 increase it.
            - If the request involves more abstract or subjective criteria (e.g., "make it look warmer"), choose settings that best fit the description based on common design principles and aesthetics.
        4. **Response Formatting:** Return your response in JSON format with details about the selected adjustment type, relevant object (if applicable), and specific options.

        **Output Format**:
        {
            "Adjustment Type": <type_of_adjustment>,
            "Object Class": <object_class>, // Only needed if applicable, e.g., for color changes
            "Adjustment Options": [<options>] 
        }

        **Examples**:  
        User Request: "Make the car look blue."
        ```json
        {
            "Adjustment Type": "recolor",
            "Object Class": "the car",
            "Adjustment Options": ["#ADD8E6", "#87CEEB", "#6495ED", "#4682B4", "#0000FF", "#0000CD", "#00008B", "#000080"]
        }
        ```
        User Request: "Increase the contrast."
        ```json
        {
            "Adjustment Type": "contrast",
            "Adjustment Options": [0.25, 0.5, 1, 1.5, 2]
        }
        ```
    """
    session['option_messages'].append(message_template('system', system_prompt))
    session['option_messages'].append(message_template('user', user_message))
    response = json.loads(api_answer(session['option_messages'], "json"))
    print(response)
    return response


def hex_to_rgb(hex_color):
    if len(hex_color) == 3:
        return hex_color
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Invalid hex color code")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return [r, g, b]


@app.route('/recolor', methods=['POST'])
def recolor():
    global prev_object_class, prev_image_data, prev_masks
    data = request.get_json()
    image_data = data['image']
    object_class = data["ObjectClass"]
    if image_data.startswith('data:image/png;base64,'):
        header, base64_data = image_data.split(',', 1)
    elif image_data.startswith('data:image/jpeg;base64,'):
        header, base64_data = image_data.split(',', 1)
    else:
        return "Invalid image format", 400
    image = base64.b64decode(base64_data)
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if object_class != prev_object_class or image_data != prev_image_data:
        masks = object_segment(image, object_class)
        masks = masks.astype(np.uint8) * 255
        prev_object_class = object_class
        prev_image_data = image_data
        prev_masks = masks
    else:
        masks = prev_masks
    target_color_rgb = hex_to_rgb(data['AdjustmentValue'])  
    target_color_hsv = cv2.cvtColor(np.uint8([[target_color_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    for mask in masks:
        new_h = np.where(mask > 0, target_color_hsv[0], h)
        s_factor = (target_color_hsv[1] / 255.0) if target_color_hsv[1] > 0 else 1.0
        v_factor = (target_color_hsv[2] / 255.0) if target_color_hsv[2] > 0 else 1.0
        new_s = np.where(mask > 0, np.clip(s * s_factor, 0, 255).astype(np.uint8), s)
        new_v = np.where(mask > 0, np.clip(v * v_factor, 0, 255).astype(np.uint8), v)

        result_hsv = cv2.merge([new_h, new_s, new_v])
        result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite("recolor.jpg", result_bgr)
    _, buffer = cv2.imencode('.jpg', result_bgr)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    response = {
        "modified_image": f"data:image/jpeg;base64,{encoded_image}"
    }

    return jsonify(response)


@app.route('/brightness', methods=['POST'])
def brightness():
    data = request.get_json()
    image_data = data['image']
    ad_value = float(data['AdjustmentValue'])
    if image_data.startswith('data:image/png;base64,'):
        header, base64_data = image_data.split(',', 1)
    elif image_data.startswith('data:image/jpeg;base64,'):
        header, base64_data = image_data.split(',', 1)
    else:
        return "Invalid image format", 400
    image = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image))
    enhancer = ImageEnhance.Brightness(image)
    image_adjusted = enhancer.enhance(ad_value)
    buffered = BytesIO()
    image_adjusted.save(buffered, format="JPEG")
    image_adjusted.save("brightness.jpg", format="JPEG")
    buffered.seek(0)
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = {
        "modified_image": f"data:image/jpeg;base64,{encoded_image}"
    }

    return jsonify(response)


@app.route('/generate_ui', methods=['POST'])
def generate_ui():
    session['ui_messages'] = []
    user_message = request.form['message']
    try:
        image = request.files['image']
    except:
        image=None
    print('image',image)
    if image:
        


        result = option(user_message)
        adjustment_type = result["Adjustment Type"]
        object_class = result.get("Object Class", None)
        adjustment_options = result["Adjustment Options"]
        system_prompt = f"""
            **Instructions**:
            You are an expert UI design assistant specialized in generating aesthetically pleasing, simple, direct,
            and user-friendly HTML pages tailored to specific image editing tasks as per user requests. 
            Your task is to generate complete, functional HTML code embedded with CSS and JavaScript based on the individual needs of each user.
    
            **Guidelines**:
            1. The layout should be intuitive and clear, facilitating quick familiarity and ease of use for users.
            2. For the operation '{adjustment_type}, decide on the most suitable form of interactive elements (such as sliders, color pickers, etc.) that best facilitate user interaction and achieve the desired effect. Use the provided adjustment options: {adjustment_options}. All necessary controls should be directly visible and accessible within the page. If a slider is generated, ensure it allows continuous adjustment by providing appropriate min, max, and step values.
            3. Adopt a minimalist and modern design style, with harmonious color schemes and avoid overly complex or harsh designs.
            4. Make sure all interactive elements (buttons, sliders, etc.) are highly clickable and responsive.
            5. Ensure the generated HTML code is well-structured, with clear comments to facilitate future maintenance.
            6. The HTML should read the image from localStorage, specifically searching for the key "uploadedImage". It should call the appropriate API endpoint based on the operation performed by the user. The API endpoint name is constructed by adding a forward slash before the adjustment type, e.g., '/{adjustment_type}'. Use the keys: image, ObjectClass ({object_class}), and AdjustmentValue, where AdjustmentValue corresponds to the value selected by the user for the operation. The API response will be in the form: response = {{ "modified_image": "data:image/jpeg;base64,{{encoded_image}}" }}, and you need to use the modified_image field to display the result.
            7. The layout should be in a **left-aligned**, chat-like format, ensuring it is not overly large, with the parent container size not exceeding **500px** but also sized adequately to accommodate the image. Only **one image** should be displayed at a time, and this image should always be shown in the same position within the layout.
            8. Provide only the HTML code as output without any additional text or explanation.
    
            Please adhere to the above guidelines to generate the HTML page code specifically for this user instruction: {user_message}
        """
        session['ui_messages'].append(message_template('system', system_prompt))
        session['ui_messages'].append(message_template('user', "generate the appropriate UI"))
        ui_response = process_symbol(api_answer(session['ui_messages']))
    else:
        response_message, html_template, session['data_agent_replacements_json']= generate_response(user_message)
        session['data_agent_messages'].append(message_template('assistant', html_template))
        ui_response = process_symbol(response_message)

    print(ui_response)
    return jsonify({'response': ui_response})


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
