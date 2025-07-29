
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from huggingface_hub import HfApi
import pyresearch

app = Flask(__name__)

# Load the YOLOv8 model with fallback to pre-trained model
try:
    model = YOLO('best.pt')
    print("YOLOv8 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv8 model 'best.pt': {e}. Falling back to yolov8n.pt")
    try:
        model = YOLO('yolov8n.pt')  # Fallback to pre-trained nano model
        print("Fallback YOLOv8 model loaded successfully")
    except Exception as e2:
        print(f"Error loading fallback YOLO model: {e2}")
        model = None

# Load Llama model and tokenizer (using only Llama-3.2-1B-Instruct with explicit token)
LLAMA_MODEL_NAME_32 = "meta-llama/Llama-3.2-1B-Instruct"  # Single model as requested
try:
    # Ensure token is available
    api = HfApi()
    token = api.token  # Retrieves token if logged in
    if not token:
        token = input("Please enter your Hugging Face token: ")  # Prompt for manual input if not logged in

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME_32, token=token)
    llm = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME_32,
        torch_dtype=torch.float16,
        device_map="cpu" if not torch.cuda.is_available() else "auto",
        token=token
    )
    print(f"Llama {LLAMA_MODEL_NAME_32} loaded successfully")
except Exception as e:
    print(f"Error loading Llama {LLAMA_MODEL_NAME_32}: {e}. Using static fallback data.")
    llm = None

# Static fallback data for classes (approximate per 100g)
static_nutritional_data = {
    'Rasam': {
        "calories": 25,
        "carbohydrates": {"total": 5.0, "sugars": 1.0, "fiber": 1.0},
        "fats": {"total": 1.0, "saturated": 0.2, "unsaturated": 0.8},
        "proteins": {"total": 1.0, "amino_acids": {"lysine": 20.0, "methionine": 10.0}},
        "vitamins": {"vitamin_a": 50.0, "vitamin_c": 5.0, "vitamin_d": 0.0},
        "minerals": {"calcium": 20.0, "iron": 1.0, "potassium": 150.0},
        "sterols": {"cholesterol": 0.0, "phytosterols": 10.0}
    },
    'Veg Kurma': {
        "calories": 120,
        "carbohydrates": {"total": 15.0, "sugars": 3.0, "fiber": 2.0},
        "fats": {"total": 8.0, "saturated": 2.0, "unsaturated": 5.0},
        "proteins": {"total": 3.0, "amino_acids": {"lysine": 40.0, "methionine": 20.0}},
        "vitamins": {"vitamin_a": 200.0, "vitamin_c": 10.0, "vitamin_d": 0.0},
        "minerals": {"calcium": 50.0, "iron": 2.0, "potassium": 300.0},
        "sterols": {"cholesterol": 0.0, "phytosterols": 15.0}
    },
    'Neer Dosa': {
        "calories": 70,
        "carbohydrates": {"total": 14.0, "sugars": 0.5, "fiber": 1.0},
        "fats": {"total": 1.0, "saturated": 0.3, "unsaturated": 0.6},
        "proteins": {"total": 2.0, "amino_acids": {"lysine": 25.0, "methionine": 15.0}},
        "vitamins": {"vitamin_a": 0.0, "vitamin_c": 0.0, "vitamin_d": 0.0},
        "minerals": {"calcium": 15.0, "iron": 0.5, "potassium": 80.0},
        "sterols": {"cholesterol": 0.0, "phytosterols": 5.0}
    },
    'Idiyappam': {
        "calories": 150,
        "carbohydrates": {"total": 30.0, "sugars": 1.0, "fiber": 1.5},
        "fats": {"total": 2.0, "saturated": 0.5, "unsaturated": 1.2},
        "proteins": {"total": 3.0, "amino_acids": {"lysine": 30.0, "methionine": 20.0}},
        "vitamins": {"vitamin_a": 0.0, "vitamin_c": 0.0, "vitamin_d": 0.0},
        "minerals": {"calcium": 20.0, "iron": 1.0, "potassium": 100.0},
        "sterols": {"cholesterol": 0.0, "phytosterols": 5.0}
    },
    'Soya Chunk Biryani': {
        "calories": 200,
        "carbohydrates": {"total": 35.0, "sugars": 2.0, "fiber": 3.0},
        "fats": {"total": 5.0, "saturated": 1.0, "unsaturated": 3.0},
        "proteins": {"total": 10.0, "amino_acids": {"lysine": 50.0, "methionine": 30.0}},
        "vitamins": {"vitamin_a": 50.0, "vitamin_c": 5.0, "vitamin_d": 0.0},
        "minerals": {"calcium": 40.0, "iron": 2.0, "potassium": 250.0},
        "sterols": {"cholesterol": 0.0, "phytosterols": 10.0}
    },
    'Dal Khichdi': {
        "calories": 120,
        "carbohydrates": {"total": 20.0, "sugars": 1.0, "fiber": 2.0},
        "fats": {"total": 3.0, "saturated": 0.5, "unsaturated": 2.0},
        "proteins": {"total": 5.0, "amino_acids": {"lysine": 40.0, "methionine": 20.0}},
        "vitamins": {"vitamin_a": 30.0, "vitamin_c": 2.0, "vitamin_d": 0.0},
        "minerals": {"calcium": 25.0, "iron": 1.5, "potassium": 200.0},
        "sterols": {"cholesterol": 0.0, "phytosterols": 5.0}
    },
    'Error': {
        "calories": 0,
        "carbohydrates": {"total": 0.0, "sugars": 0.0, "fiber": 0.0},
        "fats": {"total": 0.0, "saturated": 0.0, "unsaturated": 0.0},
        "proteins": {"total": 0.0, "amino_acids": {"lysine": 0.0, "methionine": 0.0}},
        "vitamins": {"vitamin_a": 0.0, "vitamin_c": 0.0, "vitamin_d": 0.0},
        "minerals": {"calcium": 0.0, "iron": 0.0, "potassium": 0.0},
        "sterols": {"cholesterol": 0.0, "phytosterols": 0.0}
    }
}

def get_nutritional_data_from_llama(food_name):
    """Use Llama model to generate nutritional information, fall back to static data if unavailable."""
    if llm is None:
        print(f"Llama model unavailable, using static fallback for {food_name}")
        # Case-insensitive lookup for static data
        food_name_lower = food_name.lower()
        for key in static_nutritional_data:
            if key.lower() == food_name_lower:
                return static_nutritional_data[key]
        return static_nutritional_data['Error']

    prompt = f"""
    You are a nutrition expert. Provide detailed nutritional information for a typical serving (100g or one standard portion) of {food_name}.
    Return the response in JSON format with the following structure:
    ```json
    {{
        "calories": <integer, kcal>,
        "carbohydrates": {{"total": <float, g>, "sugars": <float, g>, "fiber": <float, g>}},
        "fats": {{"total": <float, g>, "saturated": <float, g>, "unsaturated": <float, g>}},
        "proteins": {{"total": <float, g>, "amino_acids": {{"lysine": <float, mg>, "methionine": <float, mg>}}}},
        "vitamins": {{"vitamin_a": <float, IU>, "vitamin_c": <float, mg>, "vitamin_d": <float, IU>}},
        "minerals": {{"calcium": <float, mg>, "iron": <float, mg>, "potassium": <float, mg>}},
        "sterols": {{"cholesterol": <float, mg>, "phytosterols": <float, mg>}}
    }}
    ```
    Ensure values are realistic and based on standard nutritional data for {food_name}. If exact values are unknown, provide educated estimates based on similar foods. Return ONLY the JSON object inside the ```json``` block, with no additional text or explanations outside the JSON.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu" if not torch.cuda.is_available() else "auto")
        outputs = llm.generate(
            **inputs,
            max_length=500,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Raw Llama response for {food_name}: {response}")

        start = response.find("```json") + 7
        end = response.find("```", start)
        if start > 6 and end > start:
            json_str = response[start:end].strip()
            try:
                data = json.loads(json_str)
                print(f"Parsed JSON for {food_name}: {data}")
                return data
            except json.JSONDecodeError as e:
                print(f"JSON parsing error for {food_name}: {e}")
                return static_nutritional_data.get(food_name, static_nutritional_data['Error'])
        else:
            print(f"No valid JSON found in Llama response for {food_name}")
            return static_nutritional_data.get(food_name, static_nutritional_data['Error'])
    except Exception as e:
        print(f"Error with Llama model for {food_name}: {e}")
        return static_nutritional_data.get(food_name, static_nutritional_data['Error'])

def process_image(image):
    if model is None:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.putText(img, "Model unavailable", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str, [{"label": "Error", "confidence": 0.0, "box": [0, 0, 0, 0], "nutrition": get_nutritional_data_from_llama("Error")}]

    results = model(image)
    detections = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        names = result.names

        for box, score, cls in zip(boxes, scores, classes):
            if score > 0.5:
                label = names[int(cls)]
                x1, y1, x2, y2 = map(int, box)
                nutrition = get_nutritional_data_from_llama(label)
                detections.append({
                    'label': label,
                    'confidence': float(score),
                    'box': [x1, y1, x2, y2],
                    'nutrition': nutrition
                })

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = f"{det['label']} ({det['confidence']:.2f})"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str, detections

def process_video(video_path):
    if model is None:
        return [], [{"label": "Error", "confidence": 0.0, "box": [0, 0, 0, 0], "nutrition": get_nutritional_data_from_llama("Error")}]

    cap = cv2.VideoCapture(video_path)
    frames = []
    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_str, frame_dets = process_image(img)
        if img_str:
            frames.append(img_str)
            detections.extend(frame_dets)

    cap.release()
    return frames, detections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file.mimetype.startswith('image'):
        try:
            img = Image.open(file).convert('RGB')
            img_str, detections = process_image(img)
            return jsonify({'image': img_str, 'detections': detections})
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

    elif file.mimetype.startswith('video'):
        video_path = 'temp_video.mp4'
        try:
            file.save(video_path)
            frames, detections = process_video(video_path)
            os.remove(video_path)
            if not frames:
                return jsonify({'error': 'Failed to process video'}), 400
            return jsonify({'frames': frames, 'detections': detections})
        except Exception as e:
            print(f"Error processing video: {e}")
            if os.path.exists(video_path):
                os.remove(video_path)
            return jsonify({'error': f'Invalid video file: {str(e)}'}), 400

    return jsonify({'error': 'Unsupported file type'}), 400

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)  # Disable auto-reload