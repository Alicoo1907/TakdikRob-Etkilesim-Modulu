from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import os
import subprocess
import threading
import time
import cv2
import numpy as np
from collections import OrderedDict, deque

# Yerel kütüphaneleri ekle
import sys
sys.path.append(os.path.join(os.getcwd(), 'pykinect2'))
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

# Robot Durum Dosyası (Red/Green Eye Sync)
ROBOT_STATE_FILE = "robot_state.json"

# Skeleton çizim için bağlantı listesi (Kinect V2)
JOINT_CONNECTIONS = [
    (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_SpineMid),
    (PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineShoulder),
    (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_Neck),
    (PyKinectV2.JointType_Neck, PyKinectV2.JointType_Head),
    (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft),
    (PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft),
    (PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft),
    (PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft),
    (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight),
    (PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight),
    (PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight),
    (PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight)
]

app = Flask(__name__)
CORS(app)

# Global Kinect nesnesi (Streaming için)
kinect_sensor = None
streaming_active = False
automation_process = None

def generate_frames():
    global kinect_sensor
    sensor = get_kinect()
    if sensor is None:
        return
    while True:
        if sensor.has_new_color_frame():
            frame = sensor.get_last_color_frame()
            # Optimizasyon: Önce küçült sonra işle (1080p -> 480p)
            frame = frame.reshape((1080, 1920, 4))
            frame = cv2.resize(frame, (854, 480)) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Skeleton çizimi (Eğer varsa)
            if sensor.has_new_body_frame():
                bodies = sensor.get_last_body_frame()
                if bodies:
                    # En yakın vücudu bulalım
                    closest_body = None
                    min_z = float('inf')
                    for i in range(sensor.max_body_count):
                        b = bodies.bodies[i]
                        if b.is_tracked:
                            z = b.joints[PyKinectV2.JointType_SpineBase].Position.z
                            if z < min_z and z > 0.5:
                                min_z = z
                                closest_body = b
                    
                    if closest_body:
                        body = closest_body
                        # Eklemleri çiz
                        for i in range(PyKinectV2.JointType_Count):
                            joint = body.joints[i]
                            if joint.TrackingState == PyKinectV2.TrackingState_Tracked:
                                pos = sensor.body_joint_to_color_space(joint)
                                # Küçültülmüş resme göre koordinatları ayarla (854/1920, 480/1080)
                                x = int(pos.x * (854/1920))
                                y = int(pos.y * (480/1080))
                                if np.isfinite(x) and np.isfinite(y):
                                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                        
                        # Kemikleri çiz
                        for j1, j2 in JOINT_CONNECTIONS:
                            if body.joints[j1].TrackingState == PyKinectV2.TrackingState_Tracked and \
                               body.joints[j2].TrackingState == PyKinectV2.TrackingState_Tracked:
                                p1 = sensor.body_joint_to_color_space(body.joints[j1])
                                p2 = sensor.body_joint_to_color_space(body.joints[j2])
                                x1, y1 = int(p1.x * (854/1920)), int(p1.y * (480/1080))
                                x2, y2 = int(p2.x * (854/1920)), int(p2.y * (480/1080))
                                if np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2):
                                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)

def get_kinect():
    global kinect_sensor
    if kinect_sensor is None:
        try:
            # Sadece Color istediğimiz için maskeleyebiliriz ama PyKinect2 
            # bazen tam açılmayı sever.
            # Hem Renk hem de Vücut takip verisini açıyoruz
            kinect_sensor = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
        except Exception as e:
            print(f"HATA: Kinect başlatma hatası: {e}")
    return kinect_sensor


MAPPING_FILE = "mapping.json"

# Varsayılan eşleşme tablosu
default_mapping = {
    "El sallamak (NTU-23)": "EL SALLAMA",
    "Neşelenmek (NTU-22)": "KOLLARI İKİ YANA AÇMA",
    "Ceket giymek (NTU-14)": "KOLLAR ÖNDE DİRSEKLER YUKARI",
    "Bir şeye tekme atmak (NTU-24)": "PİRAMİT",
    "Tek ayak üzerinde zıplamak (NTU-26)": "N HARFİ",
    "Yukarı zıplamak (NTU-27)": "SAĞ YUMRUK",
    "Baş eğmek/Evet (NTU-35)": "SOL YUMRUK",
    "Başını sallamak/Hayır (NTU-36)": "SAĞ EL YERE PARALEL",
    "Selam durmak (NTU-38)": "ASKER SELAMI",
    "Elleri önde çaprazlamak (NTU-40)": "SARILMA (X)",
    "Düşmek (NTU-43)": "TERS PİRAMİT",
    "Yelpazelenmek (NTU-49)": "KOLLAR ÖNE UZATMA",
    "Şapka/kep takmak (NTU-20)": "KOLLAR YUKARI UZATMA",
    "Bir şeyi fırlatmak (NTU-7)": "SAĞ KOL SAĞA",
    "Bir şeyi yerden almak (NTU-6)": "SOL KOL SOLA"
}

# Dosyadan yükle / kaydet
def load_mapping():
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            # OrderedDict ile sıralamayı koru
            return json.load(f, object_pairs_hook=OrderedDict)
    else:
        return OrderedDict()

def save_mapping(data):
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_robot_state(color):
    """Robotun göz rengini (durumunu) dosyaya yazar."""
    try:
        with open(ROBOT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"eye_color": color}, f)
    except Exception as e:
        print(f"HATA: Robot durumu kaydedilemedi: {e}")

# --- Flask endpointleri ---
@app.route("/get_mapping", methods=["GET"])
def get_mapping():
    data = load_mapping()
    # jsonify yerine direkt json.dumps (OrderedDict sırasını korur)
    return app.response_class(
        response=json.dumps(data, ensure_ascii=False),
        mimetype="application/json"
    )

@app.route("/update_mapping", methods=["POST"])
def update_mapping():
    data = request.get_json()
    if not data:
        return "No mapping data received", 400
    save_mapping(data)
    print("Esleme tablosu guncellendi.")
    return "Mapping updated", 200

@app.route("/reset_mapping", methods=["POST"])
def reset_mapping():
    """mapping.json dosyasını varsayılan haline döndürür"""
    try:
        save_mapping(default_mapping)
    
        return jsonify({"status": "OK", "message": "Mapping varsayılana döndü."}), 200
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 500

@app.route("/set_eye_color", methods=["POST"])
def set_eye_color():
    """Ubuntu'dan gelen göz rengi bilgisini alır ve dosyaya yazar."""
    data = request.get_json()
    if not data or "color" not in data:
        return jsonify({"status": "ERROR", "message": "Missing 'color' field"}), 400
    
    color = data["color"].lower() # "red" or "green"
    if color not in ["red", "green"]:
         return jsonify({"status": "ERROR", "message": "Invalid color. Use 'red' or 'green'"}), 400

    save_robot_state(color)
    # Sessiz: Durum dosyası güncellendi
    return jsonify({"status": "OK", "message": f"State set to {color}"}), 200

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/run_automation", methods=["POST"])
def run_automation():
    global automation_process
    try:
        # Önceki süreci durdur (isteğe bağlı)
        if automation_process and automation_process.poll() is None:
            automation_process.terminate()

        project_dir = r"C:\Users\Takdik2\Desktop\Nao_Imitiate_Project"
        script_path = os.path.join(project_dir, "kinect_automation.py")

        # Arka planda çalıştır (Log tutmuyoruz artık)
        automation_process = subprocess.Popen(
            [sys.executable, script_path],
            cwd=project_dir
        )
        
        return jsonify({"status": "OK", "message": "Otomasyon arka planda başlatıldı"}), 200

    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 500


if __name__ == "__main__":
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    print("Windows API hazir: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)