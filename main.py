from generate import generate_motion_by_label
from angle_converter import compute_joint_angles
import os
import json
import threading
import time
import requests

# --- Ubuntu ayarları ---
ubuntuIP = 'BURAYA_UBUNTU_IP'  # Ubuntu/WSL IP adresi (ornek: 172.26.207.236)
url = f"http://{ubuntuIP}:2307/upload_msg"

# --- Windows Mapping API ---
mapping_url = "http://127.0.0.1:5000/get_mapping"

# --- Etiket haritası (Model çıktılarını isimlendirmek için) ---
label_map = {
    0: ["El sallamak (NTU-23)", "EL SALLAMA"],
    1: ["Neşelenmek (NTU-22)", "KOLLARI İKİ YANA AÇMA"],
    2: ["Ceket giymek (NTU-14)", "KOLLAR ÖNDE DİRSEKLER YUKARI"],
    3: ["Bir şeye tekme atmak (NTU-24)", "PİRAMİT"],
    4: ["Tek ayak üzerinde zıplamak (NTU-26)", "N HARFİ"],
    5: ["Yukarı zıplamak (NTU-27)", "SAĞ YUMRUK"],
    6: ["Baş eğmek/Evet (NTU-35)", "SOL YUMRUK"],
    7: ["Selam durmak (NTU-38)", "ASKER SELAMI"],
    8: ["Başını sallamak/Hayır (NTU-36)", "SAĞ EL YERE PARALEL"],
    9: ["Elleri önde çaprazlamak (NTU-40)", "SARILMA (X)"],
    10: ["Düşmek (NTU-43)", "TERS PİRAMİT"],
    11: ["Yelpazelenmek (NTU-49)", "KOLLAR ÖNE UZATMA"],
    12: ["Şapka/kep takmak (NTU-20)", "KOLLAR YUKARI UZATMA"],
    13: ["Bir şeyi fırlatmak (NTU-7)", "SAĞ KOL SAĞA"],
    14: ["Bir şeyi yerden almak (NTU-6)", "SOL KOL SOLA"]
}

def trigger_motion(user_input):
    """
    Belirli bir komut veya etiket için hareketi üretip Ubuntu'ya gönderir.
    """
    input_str = str(user_input)

    
    # --- Windows Mapping API'den mapping al ---
    mapping_data = {}
    try:
        r = requests.get(mapping_url, timeout=2)
        mapping_data = r.json()
    except Exception as e:
        pass

    # 1. Girdi sayı ise, onu önce "Ekran İsmi"ne çevir (Eşleşme tablosu için)
    display_name = input_str
    if input_str.isdigit():
        label_id = int(input_str)
        if label_id in label_map:
            display_name = label_map[label_id][0] # "El Sallama (NTU-23)" gibi
    
    # 2. Web Mapping'den geçecek komutu belirle
    # Eğer web arayüzünde "El Sallama (NTU-23)" için bir karşılık varsa onu al
    target_action = mapping_data.get(display_name, display_name)
    


    # 3. Sonuç ismini (target_action) tekrar Robot Label'ına (sayı) çevir
    label = None
    if str(target_action).isdigit():
        label = int(target_action)
    else:
        # label_map içinde tam eşleşme ara
        for k, names in label_map.items():
            # Hem Ekran İsmi ("El Sallama (NTU-23)") hem de Robot İsmi ("EL SALLAMA") ile eşleşebilir
            if any(str(target_action).strip().upper() == n.strip().upper() for n in names):
                label = k
                break
    
    if label is None:
        print(f"HATA: '{target_action}' hareketi için robot tarafında karşılık bulunamadı!")
        return False

    gesture_name = label_map.get(label, ["Bilinmeyen"])[0]
    print(f"[{gesture_name}] tanindi -> Robot: [{target_action}]")

    # --- Hareket üret (VAE) ---
    try:
        fake_np = generate_motion_by_label(label)
    except Exception as e:
        print(f"HATA: Hareket üretim hatası (generate_motion_by_label): {e}")
        return False

    # --- Joint hazırlığı ---
    joint_list = ['Center', 'ShoulderLeft', 'ElbowLeft', 'WristLeft',
                  'ShoulderRight', 'ElbowRight', 'WristRight']
    
    T = fake_np.shape[2]
    frames_as_dicts = []
    for t in range(T):
        frame = {}
        for j, joint in enumerate(joint_list):
            frame[joint] = {
                "X": float(fake_np[j, 0, t]),
                "Y": float(fake_np[j, 1, t]),
                "Z": float(fake_np[j, 2, t])
            }
        frames_as_dicts.append(frame)

    # --- Radyan açılar ---
    all_radian_angles = []
    for frame in frames_as_dicts:
        angles = compute_joint_angles(frame) 
        all_radian_angles.append(angles)

    # --- JSON oluştur ---
    joint_names = ["RShoulderPitch","RShoulderRoll","RElbowRoll","RElbowYaw",
                   "LShoulderPitch","LShoulderRoll","LElbowRoll","LElbowYaw"]
    
    frames_as_joints = []
    for angles in all_radian_angles:
        frame_dict = {joint: float(angle) for joint, angle in zip(joint_names, angles)}
        frames_as_joints.append(frame_dict)
        
    data_to_send = {
        "label": label,
        "reason": str(user_input),
        "frames": frames_as_joints
    }

    # --- Ubuntu’ya gönderim ---
    def send_to_ubuntu(payload):
        try:
            r = requests.post(url, json={"msg": payload}, timeout=30)
            print(f"Robot hareketi gonderildi.")
        except Exception as e:
            print(f"HATA: Ubuntu gönderim hatası: {e}")

    threading.Thread(target=send_to_ubuntu, args=(data_to_send,)).start()
    return True

if __name__ == "__main__":
    print("--- NAO Hareket Motoru (Manuel Mod) ---")
    u_in = input("Hareket komutu gir: ")
    trigger_motion(u_in)
