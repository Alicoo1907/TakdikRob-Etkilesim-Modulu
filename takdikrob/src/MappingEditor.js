import React, { useEffect, useState } from "react";
import { DragDropContext, Droppable, Draggable } from "@hello-pangea/dnd";
import { getMapping, updateMapping, resetMapping } from "./api";

import axios from "axios";

function MappingEditor() {
  const [mapping, setMapping] = useState({});
  const [keys, setKeys] = useState([]);
  const [values, setValues] = useState([]);
  const [videoStream, setVideoStream] = useState(null);

  useEffect(() => {
    getMapping().then((res) => {
      const data = res.data;
      setMapping(data);

      // JSON’daki orijinal sıralamayı koru
      const orderedEntries = Object.entries(data);
      setKeys(orderedEntries.map(([k]) => k));
      setValues(orderedEntries.map(([, v]) => v));
    });

    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => setVideoStream(stream))
      .catch((err) => console.error("Kamera açılamadı:", err));
  }, []);

  const onDragEnd = (result) => {
    if (!result.destination) return;
    const newValues = Array.from(values);
    const [removed] = newValues.splice(result.source.index, 1);
    newValues.splice(result.destination.index, 0, removed);
    const newMapping = {};
    keys.forEach((key, i) => (newMapping[key] = newValues[i]));
    setValues(newValues);
    setMapping(newMapping);
  };

  const handleSave = () => {
    updateMapping(mapping)
      .then(() => alert("✅ Eşleştirmeler kaydedildi"))
      .catch(() => alert("❌ Kaydetme hatası"));
  };

  const handleReset = () => {
    if (window.confirm("Tüm eşleştirmeleri varsayılan haline döndürmek istediğine emin misin?")) {
      resetMapping()
        .then(() => {
          getMapping().then((res) => {
            const data = res.data;
            setMapping(data);
            const orderedEntries = Object.entries(data);
            setKeys(orderedEntries.map(([k]) => k));
            setValues(orderedEntries.map(([, v]) => v));
            alert("✅ Eşleştirmeler varsayılan değerlere sıfırlandı.");
          });
        })
        .catch(() => alert("❌ Reset sırasında hata oluştu."));
    }
  };


  const handleRunAutomation = async () => {
    try {
      await axios.post("http://127.0.0.1:5000/run_automation");
      alert("✅ Kinect Otomasyonu (AI) yeni terminalde başlatıldı!");
    } catch (err) {
      alert("❌ Otomasyon başlatılamadı: " + err.message);
    }
  };

  return (
    <div style={mainContainer}>
      <h2 style={{ marginBottom: "40px" }}>Hareket Eşleştirme Arayüzü</h2>

      <div style={contentContainer}>
        {/* Sol taraf (hareket listesi + oklar + sağ liste) */}
        <div style={leftColumn}>
          <h3 style={{ marginBottom: "20px" }}>Tanınan ↔ Karşı Hareketler</h3>

          <DragDropContext onDragEnd={onDragEnd}>
            <Droppable droppableId="droppable">
              {(provided) => (
                <div ref={provided.innerRef} {...provided.droppableProps}>
                  {keys.map((key, i) => (
                    <div
                      key={i}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "1fr 40px 1fr", // sol, ok, sağ sabit
                        alignItems: "center",
                        gap: "20px",
                        marginBottom: "10px",
                        width: "100%",
                        maxWidth: "700px",
                      }}
                    >
                      {/* Sol kutu */}
                      <div style={{ ...boxStyle, textAlign: "center" }}>{key}</div>

                      {/* Ok */}
                      <div
                        style={{
                          fontSize: "24px",
                          color: "#007bff",
                          fontWeight: "bold",
                          textAlign: "center",
                        }}
                      >
                        →
                      </div>

                      {/* Sağ kutu (draggable) */}
                      <Draggable key={values[i]} draggableId={values[i]} index={i}>
                        {(provided, snapshot) => {
                          const isDragging = snapshot.isDragging;
                          const dragStyle = isDragging
                            ? { width: provided.draggableProps.style?.width || "auto" }
                            : {};

                          return (
                            <div
                              ref={provided.innerRef}
                              {...provided.draggableProps}
                              {...provided.dragHandleProps}
                              style={{
                                ...boxStyle,
                                ...provided.draggableProps.style,
                                ...dragStyle,
                                textAlign: "center",
                              }}
                            >
                              {values[i]}
                            </div>
                          );
                        }}
                      </Draggable>
                    </div>
                  ))}
                  {provided.placeholder}
                </div>
              )}
            </Droppable>
          </DragDropContext>

          <div style={{ display: "flex", gap: "12px", marginTop: "10px" }}>
            <button style={saveButton} onClick={handleSave}>
              Kaydet
            </button>
            <button style={resetButton} onClick={handleReset}>
              Reset
            </button>
          </div>
        </div>

        {/* Sağ taraf (kamera + run automation) */}
        <div style={rightColumn}>
          <h3>🎥 Kinect Canlı Yayın</h3>
          <img
            src="http://127.0.0.1:5000/video_feed"
            alt="Kinect Feed"
            style={cameraStyle}
            onError={(e) => {
              e.target.src = "https://via.placeholder.com/640x360?text=Kinect+Yayin+Bekleniyor...";
            }}
          />
          <button style={runButton} onClick={handleRunAutomation}>
            ▶️ Kinect Otomasyonu Başlat
          </button>
        </div>
      </div>
    </div>
  );

}

// --- Stil tanımları ---
const mainContainer = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  fontFamily: "Inter, system-ui, sans-serif",
  backgroundColor: "#f4f7f6",
  minHeight: "100vh",
  padding: "40px 20px",
};

const contentContainer = {
  display: "flex",
  flexDirection: "row", // Yan yana
  justifyContent: "flex-start", // Sola daya!
  alignItems: "flex-start",
  gap: "50px",
  width: "100%",
  maxWidth: "1600px",
};

const leftColumn = {
  flex: "0 0 550px", // Mapping sütunu sabit genişlik
  display: "flex",
  flexDirection: "column",
  padding: "20px",
  background: "white",
  borderRadius: "12px",
  boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
};

const rightColumn = {
  flex: "1", // Kamera sütunu kalan alanı alsın
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  padding: "20px",
  background: "#1a1a1a", // Kamera alanı koyu olsun
  borderRadius: "15px",
  color: "white",
};

const cameraStyle = {
  border: "3px solid #007bff",
  borderRadius: "12px",
  width: "100%", // Kapsayıcıya yayıl
  maxWidth: "854px",
  height: "auto",
  aspectRatio: "16/9",
  objectFit: "cover",
  marginBottom: "30px",
  backgroundColor: "#000",
};

const boxStyle = {
  border: "1px solid #e0e0e0",
  borderRadius: "10px",
  padding: "12px 18px",
  background: "#ffffff",
  cursor: "grab",
  fontSize: "14px",
  fontWeight: "500",
  color: "#333",
  boxShadow: "0 2px 4px rgba(0,0,0,0.02)",
  transition: "all 0.2s ease",
  minHeight: "56px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  wordBreak: "break-word",
};

const saveButton = {
  flex: 1,
  padding: "14px",
  fontSize: "16px",
  fontWeight: "bold",
  borderRadius: "10px",
  border: "none",
  background: "#007bff",
  color: "white",
  cursor: "pointer",
  transition: "background 0.3s",
};

const resetButton = {
  flex: 1,
  padding: "14px",
  fontSize: "16px",
  borderRadius: "10px",
  border: "1px solid #dc3545",
  background: "transparent",
  color: "#dc3545",
  cursor: "pointer",
};

const runButton = {
  background: "#28a745",
  color: "white",
  padding: "18px 40px",
  borderRadius: "50px",
  fontSize: "18px",
  fontWeight: "bold",
  border: "none",
  cursor: "pointer",
  boxShadow: "0 4px 15px rgba(40, 167, 69, 0.3)",
};

export default MappingEditor;
