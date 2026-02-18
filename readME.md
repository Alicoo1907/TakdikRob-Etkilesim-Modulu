# Nao Gesture Control & Action Mapping Project

Bu proje, Kinect V2 sensörü ile insan jestlerini tanıyan, tanınan jeste karşılık gelen robot hareketini **NAO-GAT** modeli ile anlık olarak üreten ve **NAO Robotu**na gönderen bir jest kontrol sistemidir. Sistem, doğrudan "birebir taklit" yerine, **"X Hareketini Algıla -> Y Hareketini Üret ve Oynat"** mantığıyla çalışır.

---

## IP Yapılandırması

Sistemi çalıştırmadan önce aşağıdaki dosyalarda IP adreslerini kendi ağınıza göre güncelleyin:

| Dosya | Değişken | Açıklama |
| :--- | :--- | :--- |
| `main.py` (satır 10) | `ubuntuIP` | Ubuntu/WSL makinesinin IP adresi |
| `ubuntu/ubuntu.py` (satır 9) | `NAO_IP` | NAO robotunun IP adresi |
| `ubuntu/ubuntu.py` (satır 14) | `WINDOWS_IP` | Windows makinesinin IP adresi |

> [!IMPORTANT]
> IP adreslerini `BURAYA_...` yazan yerlere kendi ağ bilgilerinizi yazın. WSL IP'sini öğrenmek için Ubuntu terminalinde `hostname -I` komutunu kullanabilirsiniz.

---

## Kurulum ve Çalıştırma

### 1. Ubuntu / WSL Tarafı (Robot Sunucusu)
Bu kısım robotun bağlı olduğu WSL terminalinde çalıştırılmalıdır.
```bash
cd ubuntu
python ubuntu.py
```
> [!IMPORTANT]
> `ubuntu.py` içindeki `NAO_IP` ve `WINDOWS_IP` değişkenlerinin doğru IP'lerle eşleştiğinden emin olun.

### 2. Windows API Tarafı
```bash
python windows_api.py
```
*Port 5000 üzerinden çalışır. Web arayüzü ve Kinect otomasyonu bu API'ye bağlanır.*

### 3. Web Arayüzü
```bash
cd takdikrob
npm install
npm start
```
*Varsayılan olarak `http://localhost:3000` adresinde açılır.*

### 4. Kinect Otomasyonu (Canlı Kontrol)
Web arayüzündeki **"Kinect Otomasyonu Başlat"** butonu ile tetiklenebilir veya manuel olarak:
```bash
python kinect_automation.py
```

---

## Teknik Mimari

### Çalışma Mantığı

```
[Windows]                                              [Ubuntu/WSL]
Kinect V2 -> FC-GCN -> Jest Tanıma -> Mapping          NAO Robot Kontrolü
                         |               |                   ^
                         v               v                   |
                    NTU Etiketi -> Robot Aksiyonu            |
                                      |                      |
                                      v                      |
                              NAO-GAT Hareket Üretimi -------+
                              (Açı hesaplama + JSON)
```

#### 1. Jest Algılama (Windows)
Kinect V2'den gelen 25 eklem noktası, **FC-GCN** modeline beslenir. Model, hareketi 15 NTU-60 sınıfından birine atar ve bir etiket (label) verir.

#### 2. Eşleştirme (Windows)
Algılanan etiket, `mapping.json` tablosuna bakılarak robotun yapacağı aksiyona dönüştürülür. Bu eşleşme web arayüzünden canlı olarak değiştirilebilir.

#### 3. Hareket Üretimi (Windows)
Belirlenen robot aksiyonu için **NAO-GAT** tabanlı sistem 3D koordinatlar üretir, `angle_converter.py` ile NAO eklem açılarına dönüştürülür ve JSON formatında paketlenir.

#### 4. Robot Kontrolü (Ubuntu/WSL)
Üretilen hareket verisi Ubuntu'daki Flask sunucusuna gönderilir. Ubuntu tarafı NAO'ya bağlanarak hareketi oynatır, göz renklerini (LED) kontrol eder ve durum güncellemelerini Windows'a bildirir.

---

## Robot Durum Senkronizasyonu

Sistem, robotun durumunu göz rengi ile takip eder:

| Durum | Göz Rengi | Açıklama |
| :--- | :--- | :--- |
| Tarama / Dinleme | Kırmızı | Kinect aktif, jest algılanmayı bekliyor |
| Hareket Halinde | Yeşil | Robot hareket ediyor, Kinect duraklatılıyor |

- Başlangıçta robot gözleri **kırmızı** olarak ayarlanır.
- Hareket tetiklendiğinde gözler **yeşil** olur ve Kinect buffer temizlenir.
- Hareket tamamlandığında gözler tekrar **kırmızı** olur.
- `ALAutonomousLife` servisi otomatik olarak devre dışı bırakılır (göz rengini bozmasını engellemek için).
- Durum bilgisi `robot_state.json` dosyası üzerinden Windows ve Ubuntu arasında paylaşılır.

---

## Desteklenen Jestler (15 NTU-60 Sınıfı)

| Model | NTU ID | Jest Adı |
| :--- | :--- | :--- |
| 0 | NTU-6 | Bir şeyi yerden almak |
| 1 | NTU-7 | Bir şeyi fırlatmak |
| 2 | NTU-14 | Ceket giymek |
| 3 | NTU-20 | Şapka/kep takmak |
| 4 | NTU-22 | Neşelenmek |
| 5 | NTU-23 | El sallamak |
| 6 | NTU-24 | Bir şeye tekme atmak |
| 7 | NTU-26 | Tek ayak üzerinde zıplamak |
| 8 | NTU-27 | Yukarı zıplamak |
| 9 | NTU-35 | Baş eğmek / Evet |
| 10 | NTU-36 | Başını sallamak / Hayır |
| 11 | NTU-38 | Selam durmak |
| 12 | NTU-40 | Elleri önde çaprazlamak |
| 13 | NTU-43 | Düşmek |
| 14 | NTU-49 | Yelpazelenmek |

---

## Öne Çıkan Özellikler

- **Jest Tabanlı Kontrol:** Birebir taklit yerine, tanımlı hareketlerle robotun daha stabil ve estetik hareketler yapması sağlanır.
- **En Yakın Vücut Takibi:** Ortamdaki kalabalığı veya cansız nesneleri eleyerek sadece operatöre odaklanır.
- **Oylama Sistemi (Voting):** Son 10 tahminin en az 8'i aynı olmalı ve güven %90'ın üstünde olmalı. Bu, yanlış pozitif tetiklemeleri önler.
- **Cooldown Mekanizması:** Bir hareket tetiklendikten sonra 8 saniyelik bekleme süresi uygulanır.
- **Durum Senkronizasyonu:** Robot hareket ederken Kinect otomatik olarak duraklatılır, yeni komut gönderilmez.
- **Görsel Geri Bildirim:** Web arayüzünde Kinect görüntüsü üzerine çizilen iskelet ile takibin durumu izlenebilir.
- **Canlı Mapping Değiştirme:** Web arayüzünden hangi jestin hangi robot hareketini tetikleyeceği anlık olarak değiştirilebilir.

---

## Dosya Yapısı

| Dosya / Klasör | Görev |
| :--- | :--- |
| `kinect_automation.py` | Jest tanıma motoru, oylama sistemi ve robot tetikleme köprüsü. |
| `windows_api.py` | Flask API: Video yayını, eşleştirme yönetimi ve durum senkronizasyonu. |
| `main.py` | Hareket üretimi (NAO-GAT) ve Ubuntu'ya gönderim modülü. |
| `mapping.json` | Jest -> Robot aksiyonu eşleşme tablosu (web arayüzünden düzenlenebilir). |
| `robot_state.json` | Robotun anlık durumu (red/green). Windows-Ubuntu senkronizasyonu için. |
| `kinect_processor.py` | İskelet verisini NTU formatına normalleştiren ön işleme modülü. |
| `ubuntu/ubuntu.py` | NAO robot kontrol sunucusu (hareket çalma, LED kontrolü). |
| `takdikrob/` | React tabanlı web kontrol paneli. |
| `FC-GCN/` | Jest tanıma modeli (FC-GCN) ve ağırlık dosyaları. |
| `generate.py` | NAO-GAT tabanlı hareket üretim modülü. |
| `angle_converter.py` | 3D koordinatları NAO eklem açılarına dönüştüren modül. |

---

## Gereksinimler

### Donanım
- Kinect for Windows v2 sensörü
- NAO Robotu (V5+)

### Yazılım (Windows)
- Python 3.12+
- Kinect SDK v2.0
- `torch`, `opencv-python`, `pykinect2`, `flask`, `flask-cors`, `pyyaml`, `timm`

```bash
pip install -r requirements_kinect.txt
```

### Ubuntu / WSL
- Python 3.12+
- `flask`, `flask-cors`, `qi` (libqi 3.1.5)

---

## Parametre Ayarları

| Parametre | Dosya | Değer | Açıklama |
| :--- | :--- | :--- | :--- |
| `threshold` | `kinect_automation.py` | 0.90 | Minimum güven eşiği (%90) |
| `smoothing_window` | `kinect_automation.py` | 10 | Oylama pencere boyutu |
| `vote_count` | `kinect_automation.py` | >= 8 | Tetikleme için gerekli oy sayısı |
| `cooldown` | `kinect_automation.py` | 8 sn | Tetiklemeler arası bekleme süresi |
| `window_size` | `kinect_automation.py` | 64 | Model giriş pencere boyutu (frame) |

---

## Referanslar

Robot hareket üretimi için kullanılan generatif model hakkında detaylı bilgi için: [TakdikRob](https://github.com/Alicoo1907/TakdikRob)
