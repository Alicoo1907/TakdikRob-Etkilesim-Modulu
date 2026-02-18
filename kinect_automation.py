import sys
import os
import time
import numpy as np
import torch
import yaml
from collections import OrderedDict

# Yerel kütüphaneyi (pykinect2) ekle
sys.path.append(os.path.join(os.getcwd(), 'pykinect2'))
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

# Proje modüllerini ekle
from kinect_processor import pre_normalization
from main import trigger_motion, label_map

# FC-GCN Modellerini ekle
sys.path.append(os.path.join(os.getcwd(), 'FC-GCN', 'FC-GCN'))
from fcsa_gcn import Model

# Yapılandırma
CONFIG_PATH = 'FC-GCN/FC-GCN/config/ntu60_xview.yaml'
# Best model path
WEIGHTS_PATH = 'FC-GCN/FC-GCN/work_dir/ntu60/xview/joint/best_model_withpre.pt'

# 15 Robot Hareketi için Tam NTU-60 Eşleşme Tablosu
# Her bir robot etiketi (0-14) için bir NTU karşılığı tanımlanmıştır.
# Model Çıktısı (0-14) -> NTU ID Haritası (Tahmin edilen sınıfların gerçek NTU karşılığı)
# Modelin eğitimi sırasında sınıflar (folder names) alfabetik/sayısal sıralanmıştır.
SORTED_NTU_IDS = [6, 7, 14, 20, 22, 23, 24, 26, 27, 35, 36, 38, 40, 43, 49]

# Model İndeksi (0-14) -> Robot Hareketi (0-14) Eşleşmesi
NTU_TO_NAO = {
    0: 14,   # NTU-6 (Model 0) -> SOL KOL SOLA (Yerden Alma)
    1: 13,   # NTU-7 (Model 1) -> SAĞ KOL SAĞA (Fırlatma)
    2: 2,    # NTU-14 (Model 2) -> KOLLAR ÖNDE DİRSEKLER YUKARI (Ceket Giyme)
    3: 12,   # NTU-20 (Model 3) -> KOLLAR YUKARI UZATMA (Şapka Takma)
    4: 1,    # NTU-22 (Model 4) -> KOLLARI İKİ YANA AÇMA (Neşelenmek)
    5: 0,    # NTU-23 (Model 5) -> EL SALLAMA (El Sallama)
    6: 3,    # NTU-24 (Model 6) -> PİRAMİT (Tekme Atma)
    7: 4,    # NTU-26 (Model 7) -> N HARFİ (Zıplama)
    8: 5,    # NTU-27 (Model 8) -> SAĞ YUMRUK (Yukarı Zıplama)
    9: 6,    # NTU-35 (Model 9) -> SOL YUMRUK (Baş Eğme)
    10: 8,   # NTU-36 (Model 10) -> SAĞ EL YERE PARALEL (Baş Sallama)
    11: 7,   # NTU-38 (Model 11) -> ASKER SELAMI (Asker Selamı)
    12: 9,   # NTU-40 (Model 12) -> SARILMA (X) (Elleri Çaprazlama)
    13: 10,  # NTU-43 (Model 13) -> TERS PİRAMİT (Düşme)
    14: 11   # NTU-49 (Model 14) -> KOLLAR ÖNE UZATMA (Yelpazeleme)
}
# Not: Bazı hareketler (yumruk, kollar yukarı vb.) tek bir NTU sınıfına bağlandı 
# çünkü model sağ/sol ayrımını bu veri setinde bazen karıştırabiliyor.

class KinectAutomation:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Cihaz: {self.device}")
        
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body)
        self.model = self.load_model()
        
        self.window_size = 64 # 42'den 60'a çıktı (Daha stabil ama +0.6 saniye geçikme)
        self.buffer = []
        self.last_trigger_time = 0
        self.cooldown = 8 # 4'ten 8'e çıktı (Robot hareketi bitene kadar bekle)
        self.threshold = 0.90 # Güven eşiği (0.85'ten 0.90'a çıkarıldı - Daha kararlı olması için)
        

        
        # Yumuşatma (Smoothing) için değişkenler
        self.prediction_history = []
        self.smoothing_window = 10 # 6'dan 10'a çıktı

    def load_model(self):
        print("Model yukleniyor...")
        with open(CONFIG_PATH, 'r') as f:
            args = yaml.load(f, Loader=yaml.SafeLoader)
        
        m_args = args['model_args']
        model = Model(
            num_classes=m_args['num_classes'],
            residual=m_args['residual'],
            dropout=m_args['dropout'],
            num_person=m_args['num_person'],
            graph=m_args['graph'],
            num_nodes=m_args['num_nodes'],
            input_channels=m_args['input_channels']
        )
        
        weights = torch.load(WEIGHTS_PATH, map_location=self.device)
        # Weight anahtarlarını düzelt (stgcn -> module.stgcn ise module. kısmını siler)
        new_weights = OrderedDict()
        for k, v in weights.items():
            name = k.replace("module.", "")
            new_weights[name] = v
            
        model.load_state_dict(new_weights)
        model.to(self.device)
        model.eval()
        print("Model hazir.")
        return model

    def process_frame(self, body):
        # Kinect Joint -> NTU Joint Mapping (Sadece koordinatları al)
        # Kinect V2 has 25 joints, match their order to NTU RGB+D
        joints = []
        for i in range(25):
            j = body.joints[i]
            joints.append([j.Position.x, j.Position.y, j.Position.z])
        return np.array(joints) # (25, 3)

    def run(self):
        print("\n=== KINECT OTOMASYON AKTİF ===")
        print("Kameranın önünde durduğunuzda sizi algılayacak.")
        
        # Durum dosyası yolu
        ROBOT_STATE_FILE = "robot_state.json"
        
        try:
            last_debug_time = 0
            while True:
                # --- Robot Durum Kontrolü ---
                try:
                    if os.path.exists(ROBOT_STATE_FILE):
                        with open(ROBOT_STATE_FILE, "r") as f:
                            state = yaml.load(f, Loader=yaml.SafeLoader) # json okuyabilir
                            if state.get("eye_color") == "green":
                                # Robot hareket halinde, sessiz bekle

                                # Buffer ve geçmişi temizle ki hareket bitince eski veriler karışmasın
                                self.buffer = []
                                self.prediction_history = []
                                time.sleep(0.1)
                                continue
                except Exception as e:
                    print(f"Durum okuma hatası: {e}")
                # ----------------------------

                if self.kinect.has_new_body_frame():
                    bodies = self.kinect.get_last_body_frame()
                    if bodies is not None:
                        tracked_body = None
                        min_distance = float('inf')
                        tracked_count = 0
                        
                        for i in range(0, self.kinect.max_body_count):
                            body = bodies.bodies[i]
                            if body.is_tracked:
                                tracked_count += 1
                                # SpineBase (eklem 0) derinliğine bakarak en yakını bul
                                distance = body.joints[PyKinectV2.JointType_SpineBase].Position.z
                                if distance < min_distance and distance > 0.5: # 0.5m'den yakın hatalı okumaları ele
                                    min_distance = distance
                                    tracked_body = body
                        
                        if tracked_body:
                            if len(self.buffer) == 0:
                                print(f"Kullanici algilandi ({min_distance:.1f}m)")
                            
                            joints = self.process_frame(tracked_body)
                            self.buffer.append(joints)
                            
                            
                            # Pencere dolduğunda analiz et
                            if len(self.buffer) > self.window_size:
                                self.buffer.pop(0)
                                
                                # Her 1 frame'de bir tahmin yap (pop sonrası buffer tam doluyken)
                                if len(self.buffer) == self.window_size:
                                    self.predict()
                        else:
                            # Kimse yoksa buffer'ı temizle
                            if len(self.buffer) > 0:
                                print("Kullanici ayrildi.")
                                self.buffer = []
                
                # Her 5 saniyede bir "çalışıyorum" mesajı (sadece vücut yoksa)
                if time.time() - last_debug_time > 10:
                    last_debug_time = time.time()

                time.sleep(0.01) # ~30-60 FPS
        except KeyboardInterrupt:
            print("\nOturum kapatılıyor...")
            self.kinect.close()

    def predict(self):
        # Cooldown kontrolü
        if time.time() - self.last_trigger_time < self.cooldown:
            return

        # Veriyi hazırla: (T, V, C) -> (C, T, V, M)
        # Model 1 person (M=1) beklediği için veriyi o şekilde hazırlıyoruz
        C, T, V, M = 3, self.window_size, 25, 1
        data_final = np.zeros((C, T, V, M))
        
        data_1p = np.array(self.buffer) # (60, 25, 3)
        data_1p = data_1p.transpose(2, 0, 1) # (3, 60, 25)
        
        data_final[:, :, :, 0] = data_1p # İlk kişiye Kinect verisini koy
        
        # Normalizasyon
        data_final = pre_normalization(data_final)
        
        # Tensor'a çevir (N, C, T, V, M)
        input_tensor = torch.tensor(data_final, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_6, output_11, output_25, _ = self.model(input_tensor)
            # Genelde output_25 en detaylı olandır, ntu60 için onu kullanalım
            probs = torch.softmax(output_25, dim=1)
            conf, pred = torch.max(probs, 1)
            
            label = pred.item()
            confidence = conf.item()
            
            # Tahmini geçmişe ekle
            self.prediction_history.append(label)
            if len(self.prediction_history) > self.smoothing_window:
                self.prediction_history.pop(0)

            # Geçmişteki en sık hareketi bul (Voting)
            most_common = max(set(self.prediction_history), key=self.prediction_history.count)
            vote_count = self.prediction_history.count(most_common)
            
            # Anlık tahmini yazdır
            real_ntu_id = SORTED_NTU_IDS[label]


            # Eğer son 10 tahminin 8'i aynıysa ve güven yeterliyse tetikle
            if vote_count >= 8 and confidence > self.threshold:
                # NTU sonucunu robotun 15 hareketinden birine tercüme et
                # Not: most_common (en sık tekrar eden) etiketi tetikliyoruz
                nao_label = NTU_TO_NAO.get(most_common)
                
                if nao_label is not None:
                    gesture_name = label_map.get(nao_label, [f"Label {nao_label}"])[0]
                    real_ntu_id_trigger = SORTED_NTU_IDS[most_common]
                    print(f"\nHAREKET: {gesture_name} (Guven: {confidence:.0%})")
                    
                    # Robotu tetikle
                    success = trigger_motion(nao_label)
                    if success:
                        self.last_trigger_time = time.time()
                        self.buffer = [] 
                        self.prediction_history = [] # Tetikleme sonrası temizle
                else:
                    # Tanımlı olmayan bir NTU hareketi ise sadece ekrana yaz
                    print(f"Bilgi: NTU-{real_ntu_id} algilandi ama robotta karsiligi yok (Guven: {confidence:.2f})")

if __name__ == "__main__":
    automation = KinectAutomation()
    automation.run()
