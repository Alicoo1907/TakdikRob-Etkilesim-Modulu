from flask import Flask, request
import qi
import numpy as np
import os
import json
import time
import requests

NAO_IP = "BURAYA_NAO_IP"  # NAO robot IP (ornek: 192.168.1.100)
NAO_PORT = 9559
port = 2307

# Windows Makinesi IP'si (Lütfen Kendi IP Adresinizi Girin)
WINDOWS_IP = "BURAYA_WINDOWS_IP" # Windows makinesinin IP adresi (ornek: 192.168.1.50)
WINDOWS_API_URL = f"http://{WINDOWS_IP}:5000/set_eye_color"

Bad_request = 400
status_ok = 200
internal_server_error = 500

app = Flask(__name__)

# NAO session başta None
session = None

def connect_nao():
    """NAO robotuna bağlan"""
    global session
    if session is None:
        session = qi.Session()
        try:
            session.connect(f"tcp://{NAO_IP}:{NAO_PORT}")
            print("NAO robotuna bağlanıldı")
        except RuntimeError:
            print(f"NAO robotuna bağlanılamadı: {NAO_IP}:{NAO_PORT}")
            session = None
            return False
    return True

def play_motion(frames):
    """JSON içindeki joint açılarını NAO'ya uygular (servo güvenli versiyon)"""
    global session
    if session is None:
        print("NAO session yok, hareket oynatılamaz")
        return

    # Otonom Yaşam modunu kapat (LED'leri kontrol etmesin)
    try:
        ause = session.service("ALAutonomousLife")
        if ause.getState() != "disabled":
            ause.setState("disabled")
            print("Otonom Yaşam durduruldu.")
    except:
        pass

    motion = session.service("ALMotion")
    motion.wakeUp()
    
    # wakeUp() sonrası LED'lerin oturması için bekle
    time.sleep(1.0)
    
    # wakeUp() gözleri BEYAZ yapar, hemen YEŞİL'e çevir
    try:
        leds = session.service("ALLeds")
        leds.fadeRGB("FaceLeds", "green", 0.1)
        print("DEBUG: Gözler YEŞİL (Hareket Başlıyor)")
    except Exception as e:
        print(f"LED Hatası (Yeşil): {e}")

    joint_names = ["RShoulderPitch","RShoulderRoll","RElbowRoll","RElbowYaw",
                   "LShoulderPitch","LShoulderRoll","LElbowRoll","LElbowYaw"]

    # NAO servo limitleri [rad]
    joint_limits = {
        "RShoulderPitch": (-2.0857,  2.0857),
        "RShoulderRoll":  (-0.3142,  1.3265),
        "RElbowYaw":      (-2.0857,  2.0857),
        "RElbowRoll":     ( 0.0349,  1.5446),

        "LShoulderPitch": (-2.0857,  2.0857),
        "LShoulderRoll":  (-1.3265,  0.3142),
        "LElbowYaw":      (-2.0857,  2.0857),
        "LElbowRoll":     (-1.5446, -0.0349)
    }


    def clamp(val, lo, hi):
        return max(lo, min(hi, val))
        
    def notify_windows(color):
        """Windows API'ye robot durumunu bildirir ve Göz Rengini Değiştirir"""
        # 1. Fiziksel Göz Rengini Değiştir
        if session:
            try:
                leds = session.service("ALLeds")
                if color == "red":
                    leds.fadeRGB("FaceLeds", "red", 0.1)
                elif color == "green":
                    leds.fadeRGB("FaceLeds", "green", 0.1)
            except Exception as e:
                print(f"LED hatası: {e}")

        # 2. Windows'a Bildir
        try:
            requests.post(WINDOWS_API_URL, json={"color": color}, timeout=1)
            print(f"Windows'a bildirildi: {color}")
        except Exception as e:
            print(f"Windows bildirim hatası: {e}")

    # Harekete başlarken -> YEŞİL (Meşgul)
    notify_windows("green")

    dt = 0.05
    speed_fraction = 0.25

    for t, frame in enumerate(frames):
        angles = [frame[j] for j in joint_names]

        # limit kontrolü
        for i, j in enumerate(joint_names):
            lo, hi = joint_limits[j]
            angles[i] = clamp(angles[i], lo, hi)

        print(f"Frame {t+1}/{len(frames)}: {np.round(angles, 4)}")

        motion.setAngles(joint_names, angles, speed_fraction)
        time.sleep(dt)

    # Hareket bitince -> KIRMIZI (Hazır)
    notify_windows("red")

@app.route('/upload_msg', methods=['POST'])
def upload_msg():
    data = request.get_json()
    if data is None:
        return 'No data received', Bad_request

    motion_data = data.get('msg', None)
    if motion_data is None:
        return 'No motion data received', Bad_request

    label = motion_data.get("label")
    reason = motion_data.get("reason")
    frames = motion_data.get('frames', [])

    print(f"Gelen label: {label}, sebep: {reason}")
    print(f"✔ Toplam frame: {len(frames)}")

    if connect_nao():
        play_motion(frames)

    return 'Motion JSON received and played', status_ok

if __name__ == '__main__':
    print(f"NAO Flask sunucusu başlatıldı, port: {port}")
    
    # Başlangıçta robotun gözlerini KIRMIZI yap (Hazır/Dinliyor)
    if connect_nao():
        try:
            leds = session.service("ALLeds")
            leds.fadeRGB("FaceLeds", "red", 0.1)
            print("Başlangıç durumu: Gözler KIRMIZI yapıldı.")
        except Exception as e:
            print(f"Başlangıç LED ayarı hatası: {e}")

    app.run(host='0.0.0.0', port=port)
