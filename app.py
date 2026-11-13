import gradio as gr
import joblib
import pandas as pd

# Model & urutan fitur hasil training di Google Colab
model = joblib.load("model_rf.joblib")
fitur = joblib.load("fitur.joblib")


def prediksi(tipe, suhu_udara, suhu_proses, kecepatan, torsi, keausan):
    data = pd.DataFrame([{
        "Air temperature (K)": suhu_udara,
        "Process temperature (K)": suhu_proses,
        "Rotational speed (rpm)": kecepatan,
        "Torque (Nm)": torsi,
        "Tool wear (min)": keausan,
        "Type_L": 1 if tipe == "L" else 0,
        "Type_M": 1 if tipe == "M" else 0,   # tipe H = keduanya 0
    }])[fitur]

    p_cacat = float(model.predict_proba(data)[0][1])
    hasil = {"CACAT / GAGAL": p_cacat, "NORMAL": 1 - p_cacat}
    status = "⚠️ Mesin berpotensi GAGAL" if p_cacat >= 0.5 else "✅ Mesin NORMAL"
    return status, hasil


demo = gr.Interface(
    fn=prediksi,
    inputs=[
        gr.Radio(["L", "M", "H"], value="L", label="Tipe Produk (Low/Medium/High)"),
        gr.Number(value=300.0, label="Air temperature (K)"),
        gr.Number(value=310.5, label="Process temperature (K)"),
        gr.Number(value=1400, label="Rotational speed (rpm)"),
        gr.Number(value=60.0, label="Torque (Nm)"),
        gr.Number(value=220, label="Tool wear (min)"),
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Label(label="Probabilitas"),
    ],
    title="Prediksi Kegagalan Mesin (Predictive Maintenance)",
    description="Model Random Forest dilatih dengan dataset AI4I 2020. "
                "Masukkan kondisi mesin lalu klik Submit.",
    examples=[
        ["L", 300.0, 310.5, 1400, 60.0, 220],
        ["M", 298.1, 308.6, 1551, 42.8, 0],
    ],
)

if __name__ == "__main__":
    demo.launch()
