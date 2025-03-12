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
from lama_inpaint import inpaint_img_with_lama
from stable_diffusion_inpaint import fill_img_with_sd
from utils import *
from sam_segment import predict_masks_with_sam
from read_files import *

prev_object_class = None
prev_image_data = None
prev_masks = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'  # Use filesystem-based sessions
Session(app)
app.secret_key = 'secret_key'  # 用于启用 flash() 方法发送消息
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# GroundingDINO+SAM configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GROUNDING_DINO_CONFIG_PATH = r"/home/ec2-user/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH =r"pretrained_models/groundingdino_swint_ogc.pth"
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = r"pretrained_models/sam_vit_h_4b8939.pth"
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                             model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,device=DEVICE)
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.35
NMS_THRESHOLD = 0.82

# LAMA configuration
LAMA_CONFIG = "./lama/configs/prediction/default.yaml"
LAMA_CKPT = "./pretrained_models/big-lama"


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
@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查请求中是否包含文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    files = request.files.getlist('file')  # 获取多个文件
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No selected files'}), 400

    uploaded_paths = []
    try:
        # 处理每个上传的文件
        for file in files:
            if file and file.filename:
                # 使用 secure_filename 确保文件名安全
                from werkzeug.utils import secure_filename
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                uploaded_paths.append(file_path)

        # 返回成功的响应，包括上传的文件路径数组和文件数量
        response = {
            'message': 'Files uploaded successfully',
            'file_paths': uploaded_paths,
            'count': len(uploaded_paths)
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

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
    result = chat_single(message)
    return result


@app.before_request
def initialize_session():
    if 'option_messages' not in session:
        session['option_messages'] = []
        # if 'ui_messages' not in session:
    #     session['ui_messages'] = []

    session['ui_messages'] = []
    session['pure_text'] = []
    session['data_agent_messages'] = []  # Initialize an empty list for messages
    session['data_agent_replacements_json'] = []  # Initialize an empty list for messages
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
        The allowable operations are defined in the following list: ["recolor", "brightness", "contrast", "saturation", "object_manipuate"]. 
        Your task is to interpret the user's request and select the appropriate operation type from this predefined list, 
        then provide corresponding options.

        **Guidelines**:
        1. **Identify the Request Type:** Based on the user's request, determine which operation type from the predefined list best matches the user's need.
        2. **Object Identification:** For all operations, identify the object within the image that the user wants to modify or manipulate.
        3. **Provide Adjustment Options:**
            - For **recolor** operations, follow previous instructions for identifying objects and provide a gradient of at least 8 hexadecimal color codes that represent variations from light to dark shades of that color.
            - For **object_manipulate** operations:
                - If the action is remove, set "Is Remove" to true and confirm the object that needs to be removed.
                - If the action is relocate, leave "Adjustment Options" empty.
            - For adjustments like **brightness**, **contrast**, or **saturation**, provide adjustment values as floating-point numbers where 1 represents the original effect. Values less than 1 indicate a reduction in intensity, while values greater than 1 increase it.
            - If the request involves more abstract or subjective criteria (e.g., "make it look warmer"), choose settings that best fit the description based on common design principles and aesthetics.
        4. **Describe Possible UI Effects and Controls:** Describe potential UI elements and interactions that could be used to facilitate the selected operation. Focus on describing common UI controls such as sliders, color pickers, drag-and-drop functionality, and confirmation dialogs. Highlight how these controls can enhance the user experience by allowing real-time previews and easy manipulation of image elements.
        5. **Response Formatting:** Return your response in JSON format with details about the selected adjustment type, relevant object (if applicable), and specific options.

        **Output Format**:
        {
            "Adjustment Type": <type_of_adjustment>,
            "Object Class": <object_class>, 
            "Adjustment Options": [<options>],
            "Is Remove": <boolean>, // Only applicable for 'remove' under 'object_manipulate'
            "UI Description": <description_of_possible_ui>
        }

        **Examples**:  
        User Request: "Make the car look blue."
        ```json
        {
            "Adjustment Type": "recolor",
            "Object Class": "the car",
            "Adjustment Options": ["#ADD8E6", "#87CEEB", "#6495ED", "#4682B4", "#0000FF", "#0000CD", "#00008B", "#000080"],
            "UI Description": "The UI includes a color picker or slider that allows users to select different shades of blue. Users can see live previews of how each shade affects the car."
        }
        ```
        User Request: "Remove the tree."
        ```json
        {
            "Adjustment Type": "object_manipulate",
            "Object Class": "tree",
            "Is Remove": true,
            "UI Description": "A confirmation dialog appears to ensure the user intends to remove the tree. Once confirmed, the tree is removed from the image preview."
        }
        ```
        User Request: "Move the chair."
        ```json
        {
            "Adjustment Type": "object_manipulate",
            "Object Class": "chair",
            "Adjustment Options": [],
            "UI Description": "The UI offers drag-and-drop functionality or coordinate input fields for precise positioning. Users can move the chair around the image in real-time, seeing immediate updates in the preview."
        }
        ```
    """
    session['option_messages'].append(message_template('system', system_prompt))
    session['option_messages'].append(message_template('user', user_message))
    response = chat_single(session['option_messages'], "json")
    try:
        response = json.loads(response)
    except:
        print(response, 'json error')
    print(response)
    return response


@app.route('/remove', methods=['POST'])
def remove():
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
    image_pil = Image.open(BytesIO(image))
    image_cv = np.frombuffer(image, np.uint8)
    image_cv = cv2.imdecode(image, cv2.IMREAD_COLOR)

    masks = object_segment(image_cv, object_class)
    masks = masks.astype(np.uint8) * 255
    dilate_masks = [dilate_mask(mask, 15) for mask in masks]

    for idx, mask in enumerate(dilate_masks):
        image_pil = inpaint_img_with_lama(
            image_pil, mask, LAMA_CONFIG, LAMA_CKPT, device=DEVICE)
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    image_pil.save("remove.jpg", format="JPEG")
    buffered.seek(0)
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = {
        "modified_image": f"data:image/jpeg;base64,{encoded_image}"
    }

    return jsonify(response)


def process_image(image_data, object_class, point_coords=None, is_remove=False):
    if image_data.startswith('data:image/png;base64,'):
        header, base64_data = image_data.split(',', 1)
    elif image_data.startswith('data:image/jpeg;base64,'):
        header, base64_data = image_data.split(',', 1)
    else:
        return "Invalid image format", 400

    image = base64.b64decode(base64_data)
    image_pil = load_img_to_array(image)
    image_cv = np.frombuffer(image, np.uint8)
    image_cv = cv2.imdecode(image_cv, cv2.IMREAD_COLOR)

    if point_coords:
        masks, _, _ = predict_masks_with_sam(predictor=sam_predictor, img=image_pil, point_coords=[point_coords],
                                             point_labels=[1])
    elif object_class:
        masks = object_segment(image_cv, object_class)

    masks = masks.astype(np.uint8) * 255
    dilate_masks = [dilate_mask(mask, 15) for mask in masks]

    for idx, mask in enumerate(dilate_masks):
        image_pil = inpaint_img_with_lama(
            image_pil, mask, LAMA_CONFIG, LAMA_CKPT, device=DEVICE)

    completed_image = Image.fromarray(image_pil.astype(np.uint8))
    buffered = BytesIO()
    completed_image.save(buffered, format="JPEG")
    completed_image.save("process.jpg", format="JPEG")
    buffered.seek(0)
    completed_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    if not is_remove:
        masked_image = apply_mask_with_transparency(image_cv, masks[0])
        mask_image = Image.fromarray(masked_image, 'RGBA')
        buffered = BytesIO()
        mask_image.save(buffered, format="PNG")
        buffered.seek(0)
        mask_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            "modified_image": f"data:image/jpeg;base64,{completed_image}",
            "mask_image": f"data:image/png;base64,{mask_data}"
        }
    else:
        return {
            "modified_image": f"data:image/jpeg;base64,{completed_image}"
        }


@app.route('/object_manipulate', methods=['POST'])
def object_manipulate():
    data = request.get_json()
    image_data = data['image']
    if "ObjectClass" in data:
        object_class = data["ObjectClass"]
    else:
        object_class = None
    if "PointCoord" in data:
        point_coords = data["PointCoord"]
        print(point_coords)
    else:
        point_coords = None
    is_remove = data["IsRemove"]
    response = process_image(image_data, object_class, point_coords, is_remove)
    return jsonify(response)


@app.route('/replace', methods=['POST'])
def replace():
    data = request.get_json()
    image_data = data['image']
    object_class = data["ObjectClass"]
    text_prompt = data["TextPrompt"]
    if image_data.startswith('data:image/png;base64,'):
        header, base64_data = image_data.split(',', 1)
    elif image_data.startswith('data:image/jpeg;base64,'):
        header, base64_data = image_data.split(',', 1)
    else:
        return "Invalid image format", 400
    image = base64.b64decode(base64_data)
    image_pil = Image.open(BytesIO(image))
    image_cv = np.frombuffer(image, np.uint8)
    image_cv = cv2.imdecode(image, cv2.IMREAD_COLOR)
    masks = object_segment(image_cv, object_class)
    masks = masks.astype(np.uint8) * 255
    for idx, mask in enumerate(masks):
        image_pil = fill_img_with_sd(
            image_pil, mask, text_prompt, device=DEVICE)
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    image_pil.save("replace.jpg", format="JPEG")
    buffered.seek(0)
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = {
        "modified_image": f"data:image/jpeg;base64,{encoded_image}"
    }

    return jsonify(response)


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
    ad_value = float(data['AdjustmentValue'])
    if image_data.startswith('data:image/png;base64,'):
        header, base64_data = image_data.split(',', 1)
    elif image_data.startswith('data:image/jpeg;base64,'):
        header, base64_data = image_data.split(',', 1)
    else:
        return "Invalid image format", 400

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    final_image = image_pil.copy()
    for mask in masks:
        mask_pil = Image.fromarray(mask).convert("L")
        image_masked = image_pil.copy()
        enhancer = ImageEnhance.Brightness(image_masked)
        image_adjusted = enhancer.enhance(ad_value)
        final_image = Image.composite(image_adjusted, image_pil, mask_pil)
    buffered = BytesIO()
    final_image.save(buffered, format="JPEG")
    final_image.save("brightness.jpg", format="JPEG")
    buffered.seek(0)
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = {
        "modified_image": f"data:image/jpeg;base64,{encoded_image}"
    }

    return jsonify(response)


@app.route('/contrast', methods=['POST'])
def contrast():
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
    ad_value = float(data['AdjustmentValue'])
    if image_data.startswith('data:image/png;base64,'):
        header, base64_data = image_data.split(',', 1)
    elif image_data.startswith('data:image/jpeg;base64,'):
        header, base64_data = image_data.split(',', 1)
    else:
        return "Invalid image format", 400

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    final_image = image_pil.copy()
    for mask in masks:
        mask_pil = Image.fromarray(mask).convert("L")
        image_masked = image_pil.copy()
        enhancer = ImageEnhance.Contrast(image_masked)
        image_adjusted = enhancer.enhance(ad_value)
        final_image = Image.composite(image_adjusted, image_pil, mask_pil)
    buffered = BytesIO()
    final_image.save(buffered, format="JPEG")
    final_image.save("brightness.jpg", format="JPEG")
    buffered.seek(0)
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = {
        "modified_image": f"data:image/jpeg;base64,{encoded_image}"
    }

    return jsonify(response)


@app.route('/saturation', methods=['POST'])
def saturation():
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
    ad_value = float(data['AdjustmentValue'])
    if image_data.startswith('data:image/png;base64,'):
        header, base64_data = image_data.split(',', 1)
    elif image_data.startswith('data:image/jpeg;base64,'):
        header, base64_data = image_data.split(',', 1)
    else:
        return "Invalid image format", 400

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    final_image = image_pil.copy()
    for mask in masks:
        mask_pil = Image.fromarray(mask).convert("L")
        image_masked = image_pil.copy()
        enhancer = ImageEnhance.Color(image_masked)
        image_adjusted = enhancer.enhance(ad_value)
        final_image = Image.composite(image_adjusted, image_pil, mask_pil)
    buffered = BytesIO()
    final_image.save(buffered, format="JPEG")
    final_image.save("brightness.jpg", format="JPEG")
    buffered.seek(0)
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = {
        "modified_image": f"data:image/jpeg;base64,{encoded_image}"
    }

    return jsonify(response)


@app.route('/generate_ui', methods=['POST'])
def generate_ui():
    session['ui_messages'] = []
    data = request.get_json()
    user_message = data['message']
    query_file = 'text'
    file_info = None
    try:
        file_info = data['files']
        print(file_info)
        if 'image' in file_info[0]['type']:
            query_file = 'image'

        else:
            query_file = 'csv'

        # print('image',image)
    except:
        query_file = None
    if query_file == 'image':
        result = option(user_message)
        adjustment_type = result["Adjustment Type"]
        object_class = result.get("Object Class", None)
        if "Adjustment Options" in result:
            adjustment_options = result["Adjustment Options"]
        else:
            adjustment_options = None
        is_remove = result.get("is_remove", False)
        ui_effect = result["UI Description"]
        system_prompt = f"""
            **Instructions**:
            You are an expert UI design assistant specialized in generating aesthetically pleasing, simple, direct,
            and user-friendly HTML pages tailored to specific image editing tasks as per user requests. 
            Your task is to generate complete, functional HTML code embedded with CSS and JavaScript based on the individual needs of each user.

            **Guidelines**:
            1. The layout should be intuitive and clear, facilitating quick familiarity and ease of use for users.
            2. For the operation '{adjustment_type}', decide on the most suitable form of interactive elements (such as sliders, color pickers, etc.) that best facilitate user interaction and achieve the desired effect. Avoid using dropdowns and text inputs as much as possible to ensure all elements are directly visible and intuitive to use. Use the provided adjustment options: {adjustment_options}. All necessary controls should be directly visible and accessible within the page. If a slider is generated, ensure it allows continuous adjustment by providing appropriate min, max, and step values.
            3. Adopt a minimalist and modern design style, with harmonious color schemes and avoid overly complex or harsh designs.
            4. Make sure all interactive elements (buttons, sliders, etc.) are highly clickable and responsive.
            5. Ensure the generated HTML code is well-structured, with clear comments to facilitate future maintenance.
            6. The HTML should read the image from localStorage, specifically searching for the key "uploadedImage". It should call the appropriate API endpoint based on the operation performed by the user. The API endpoint name is constructed by adding a forward slash before the adjustment type, e.g., '/{adjustment_type}'. Use the keys: image, ObjectClass ({object_class}), IsRemove ({is_remove})and AdjustmentValue, where AdjustmentValue corresponds to the value selected by the user for the operation. The API response will be in the form: `response = {{"modified_image": "data:image/jpeg;base64,{{encoded_image}}", "mask_image": "data:image/png;base64,{{mask_data}}" (if applicable)}}. Handle cases where the 'mask_image' field may or may not be present. If 'mask_image' is available, display it appropriately overlaying the modified image.
            7. Incorporate the provided **UI Description**: {ui_effect} to understand how the API results should be displayed and interacted with in the UI. This description may include details about how the result should be previewed or additional interactions required after applying the adjustment.
            8. The layout should be in a **left-aligned**, chat-like format, ensuring it is not overly large, with the parent container size not exceeding **500px** but also sized adequately to accommodate the image. Only **one image** should be displayed at a time, and this image should always be shown in the same position within the layout.
            9. Provide only the HTML code as output without any additional text or explanation.
            Notice: please be sure to have following code to show image:
            const image = localStorage.getItem('uploadedImage');
            if (image) {{
                document.getElementById('modifiedImage').src = image;
            }}
            """
        session['ui_messages'].append(message_template('system', system_prompt))
        session['ui_messages'].append(message_template('user', user_message))
        ui_response = [process_symbol(chat_single(session['ui_messages']))]
        var_template = """
{adjustment_type}	recolor
{adjustment_options}	['#98FB98', '#7CFC00', '#00FF00', '#32CD32', '#008000', '#006400', '#228B22', '#006400']
{object_class}	hair
{is_remove}	False
{ui_effect}	The UI includes a color picker or slider that allows users to select different shades of green. Users can see live previews of how each shade affects the hair.
{user_message}	change hair color to green
        """
        print("system_prompt", system_prompt)
        # session['ui_messages'].append(message_template('system', system_prompt))
        # session['ui_messages'].append(message_template('user', "generate the appropriate UI"))
        # ui_response = process_symbol(chat_single(session['ui_messages']))
    elif query_file == 'csv':

        file_list = []
        for file in file_info:
            file_list.append('uploads/'+file['file_path'])
        data_got,data_status = iterative_agent(user_message, file_list)
        if data_status:
            ui_response = [html_generate_agent_modified(data_got, user_message)]
        else:
            ui_response=[data_got]

    else:
        session['pure_text'].append(message_template('system', ''))
        session['pure_text'].append(message_template('user', user_message))
        ui_response = [process_symbol(chat_single(session['pure_text']))]
    return jsonify({'response': ui_response})


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6060')
