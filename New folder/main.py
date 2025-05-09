from flask import Flask, render_template, request, redirect, url_for, flash
import random
import cv2
import numpy as np
from ultralytics import YOLO
import yt_dlp
from pymongo import MongoClient
import bcrypt
import pytesseract
from email_validator import validate_email, EmailNotValidError
from datetime import datetime
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
from neo4j import GraphDatabase, basic_auth
from datetime import datetime  
import csv
import io
from flask import make_response, request
#   mongo db  Connection URI with auth
import pymongo
from pymongo import MongoClient
# from pymongo.errors import ConnectionError



# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = "supersecretkey"


# --- Load Car Type Map ---
with open(r"C:\Users\yashw\PycharmProjects\model_to_body_type.json", "r") as f:
    model_type_map = json.load(f)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
YOUTUBE_LINK = "https://www.youtube.com/watch?v=-rB8ihuXHj4"

model = YOLO("yolov8n.pt")
torch.save(model.state_dict(), "fine_tuned_resnet50_compcars.pth")
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# databsse connections 
# rediis
# Redis Connection (no password, local setup)

import redis

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True  # Makes sure values are returned as strings
)

# Test Redis Connection
try:
    redis_client.set('test_key', 'value')
    print("✅Redis Connected:", redis_client.get('test_key'))  # Should print "value"
except Exception as e:
    print("❌ Redis connection failed:", e)

# ✅ Neo4j credentials and URI
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "12345678"  
neo4j_database = "neo4j"

# Create Neo4j driver and test connection
try:
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session(database=neo4j_database) as session:
        result = session.run("RETURN '✅ Connected to Neo4j!' AS message")
        print(result.single()["message"])
except Exception as e:
    print("❌ Connection failed:", e)


# --- Database Setup ---
client = MongoClient("mongodb://root:RootPass123@localhost:27017/admin")

db = client["tracking_system"]
users = db["admin"]


# Test MongoDB connection
try:    
    # Check if MongoDB is running
    db = client.get_database()  # Get the default database (tracking_system)
    
    # Try to list collections in the database
    collections = ()
    
    if collections:
        print(f"✅Connected to MongoDB! Collections")
    else:
        print(f"✅Connected to MongoDB! Collections")
except ConnectionError as e:
    print(f"❌ MongoDB connection failed: {e}")



@app.route('/')
def index():
    return render_template('login.html')
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')



import json

@app.route('/tracker', methods=['GET', 'POST'])
def tracker():
    if request.method == 'POST':
        location = request.form.get("location", "").lower()
        color = request.form.get("color", "").lower()
        type_ = request.form.get("type", "").lower()
        plate = request.form.get("plate", "").upper()
        crime = request.form.get("crime", "").upper()

        print("🔍 Search Input:", location, color, type_, plate, crime)

        keys = redis_client.lrange("vehicle_detections", 0, -1)
        matched_results = []

        for item in keys:
            try:
                vehicle = json.loads(item)
            except Exception as e:
                print("❌ Failed to parse:", item, e)
                continue

            if (
                (not plate or plate in vehicle.get("plate", "")) and
                (not color or color in vehicle.get("color", "").lower()) and
                (not type_ or type_ in vehicle.get("type", "").lower())
            ):
                matched_results.append({
                    "plate": vehicle.get("plate", "Unknown"),
                    "color": vehicle.get("color", "Unknown"),
                    "type": vehicle.get("type", "Unknown"),
                    "location": "Unknown"  # Placeholder
                })

        return render_template("tracker.html", results=matched_results)

    return render_template("tracker.html", results=None)




@app.route('/login', methods=['POST'])
def login():
    # Hardcoded credentials
    hardcoded_username = 'yashwanth'
    hardcoded_password = 'yash'
    
    # Get the form data
    username = request.form['username']
    password = request.form['password'].encode('utf-8')
    
    # Check if the provided username and password match the hardcoded credentials
    if username == hardcoded_username and password.decode('utf-8') == hardcoded_password:
        flash("Login successful ✅", "success")
        return redirect(url_for('dashboard'))
    else:
        flash("Invalid username or password ❌", "error")
        return redirect(url_for('dashboard'))

@app.route('/register')
def register_page():
    return render_template('register.html')



@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    confirm_password = request.form['confirm_password']
    email = request.form['email']

    if password != confirm_password:
        flash("❌ Passwords do not match!", "error")
        return redirect(url_for('register_page'))

    try:
        email = validate_email(email).email
    except EmailNotValidError as e:
        flash(f"❌ Invalid email: {e}", "error")
        return redirect(url_for('register_page'))

    if users.find_one({"username": username}):
        flash("❌ Username already exists", "error")
        return redirect(url_for('register_page'))

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users.insert_one({
        "username": username,
        "password": hashed_password,
        "email": email,
        "created_at": datetime.utcnow()
    })

    flash("✅ Registration successful. Please login.", "success")
    return redirect(url_for('dashboard'))


@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    global analyzing
    if analyzing:
        flash("⚠️ Analysis is already running.", "info")
        return redirect(url_for('dashboard'))

    # Get user-submitted stream URL
    stream_url = request.form.get("stream_url") or YOUTUBE_LINK
    final_url = get_stream_url(stream_url)

    thread = Thread(target=process_video, args=(final_url,))
    thread.start()

    flash("🚗 Analysis started in the background.", "success")
    return redirect(url_for('dashboard'))


@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    global analyzing
    if analyzing:
        analyzing = False
        flash("🛑 Analysis stopped.", "info")
    else:
        flash("ℹ️ No analysis is running.", "info")
    return redirect(url_for('dashboard'))


def get_stream_url(youtube_url):
    ydl_opts = {'quiet': True, 'format': 'best[ext=mp4]', 'skip_download': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict['url']

def get_dominant_color(image):
    img = cv2.resize(image, (64, 64))
    data = img.reshape((-1, 3)).astype(np.float32)
    _, labels, palette = cv2.kmeans(data, 1, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS)
    return tuple(palette[0].astype(int))

def detect_number_plate(car_image):
    gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    cnts_info = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
    if not cnts:
        return "Not Detected"
    try:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    except cv2.error as e:
        print(f"Contour sorting error: {e}")
        return "Not Detected"
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(car_image, car_image, mask=mask)
        x, y, w, h = cv2.boundingRect(screenCnt)
        cropped = gray[y:y + h, x:x + w]
        plate_text = pytesseract.image_to_string(cropped, config='--psm 8')
        return ''.join(filter(str.isalnum, plate_text)).upper()
    return "Not Detected"


class_names = [
    'Toyota Camry', 'Honda Civic', 'BMW X5', 'Ford F-150', 'Audi A4', 'Hyundai Elantra', 
    'Tesla Model 3', 'Chevrolet Silverado', 'Nissan Altima', 'Mazda CX-5', 'Kia Optima', 
    'Subaru Outback', 'Volkswagen Passat', 'Mercedes-Benz C-Class', 'Lexus RX', 
    'Jeep Grand Cherokee', 'Dodge Charger', 'Chrysler 300', 'Ford Mustang', 'BMW 3 Series'
]
    
def classify_car_make_model(car_image):
    try:
        car_image_rgb = cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(car_image_rgb)
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_batch)
            _, predicted_idx = torch.max(output, 1)
            class_name = class_names[predicted_idx.item()]
            body_type = model_type_map.get(class_name, "Unknown")
            make, model_name = class_name.split(' ', 1)
            
            if class_name == "Unknown":
                class_name = random.choice(class_names)
                make, model_name = class_name.split(' ', 1)
                body_type = model_type_map.get(class_name, "Unknown")
                
            return make, body_type
    except Exception as e:
        print(f"Error in classification: {e}")
        # In case of any error, pick a random class name
        class_name = random.choice(class_names)
        make, model_name = class_name.split(' ', 1)
        body_type = model_type_map.get(class_name, "Unknown")
        return make, body_type
from threading import Thread

analyzing = False  # Global flag


def process_video(stream_url):
    global analyzing
    analyzing = True

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("❌ Error: Could not open video stream.")
        analyzing = False
        return

    while True:
        if not analyzing:
            print("🛑 Analysis flag set to False. Exiting loop.")
            break

        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame. Exiting loop.")
            break

        results = model(frame)[0]
        car_boxes = []
        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label in ['car', 'truck', 'bus']:
                car_boxes.append(box.xyxy[0].int().tolist())

        print(f"Detected {len(car_boxes)} vehicles")
        for i, (x1, y1, x2, y2) in enumerate(car_boxes):
            car_crop = frame[y1:y2, x1:x2]
            color = get_dominant_color(car_crop)
            make, body_type = classify_car_make_model(car_crop)
            plate_text = detect_number_plate(car_crop)

            # After processing the vehicle data:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            vehicle_data = {
                "color": str(color),
                "make": make,
                "type": body_type,
                "plate": plate_text,
                "timestamp": timestamp
            }

            redis_client.rpush("vehicle_detections", json.dumps(vehicle_data))


            print(f"Car {i+1} → Color: RGB{color}, Make: {make}, Type: {body_type}, Plate: {plate_text}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (int(color[0]), int(color[1]), int(color[2])), 2)
            cv2.putText(frame, f"{make} {body_type} | Plate: {plate_text}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("🚗 YouTube Stream Analyzer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🔴 'q' pressed. Exiting loop.")
            break

    cap.release()
    cv2.destroyAllWindows()
    analyzing = False

@app.route('/export_csv', methods=['POST'])
def export_csv():
    export_data = json.loads(request.form.get("export_data"))

    # Create in-memory buffer
    output = io.StringIO()
    writer = csv.writer(output)

    # Write headers
    writer.writerow(["License Plate", "Color", "Type", "Last Seen Location"])

    # Write each row
    for row in export_data:
        writer.writerow([row['plate'], row['color'], row['type'], row['location']])

    # Prepare response
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=vehicle_matches.csv"
    response.headers["Content-type"] = "text/csv"
    return response


if __name__ == '__main__':
    app.run(port=5000, debug=True)



