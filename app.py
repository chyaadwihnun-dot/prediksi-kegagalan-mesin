import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Kegagalan Mesin", page_icon="🛠️", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model_rf.joblib")
    fitur = joblib.load("fitur.joblib")
    return model, fitur


try:
    model, fitur = load_artifacts()
except FileNotFoundError:
    st.error(
        "File model_rf.joblib dan fitur.joblib tidak ditemukan. "
        "Letakkan keduanya di folder yang sama dengan app.py."
    )
    st.stop()

st.title("🛠️ Prediksi Kegagalan Mesin")
st.caption(
    "Predictive maintenance memakai Random Forest pada dataset AI4I 2020. "
    "Masukkan kondisi sensor mesin, lalu klik Prediksi."
)

col1, col2 = st.columns(2)
with col1:
    tipe = st.selectbox("Tipe produk (Type)", ["L", "M", "H"],
                        help="L = Low, M = Medium, H = High quality")
    air_temp = st.number_input("Air temperature (K)", 290.0, 310.0, 300.0, 0.1)
    proc_temp = st.number_input("Process temperature (K)", 300.0, 320.0, 310.0, 0.1)
with col2:
    rpm = st.number_input("Rotational speed (rpm)", 1000, 3000, 1500, 10)
    torque = st.number_input("Torque (Nm)", 3.0, 80.0, 40.0, 0.1)
    wear = st.number_input("Tool wear (min)", 0, 300, 100, 1)

if st.button("Prediksi", type="primary", use_container_width=True):
    # Nama kolom disamakan dengan saat training (tanda [] diganti ())
    data = {
        "Air temperature (K)": air_temp,
        "Process temperature (K)": proc_temp,
        "Rotational speed (rpm)": rpm,
        "Torque (Nm)": torque,
        "Tool wear (min)": wear,
        "Type_L": 1 if tipe == "L" else 0,
        "Type_M": 1 if tipe == "M" else 0,
    }
    # Urutan kolom mengikuti fitur.joblib agar sama persis dengan training
    X_input = pd.DataFrame([data]).reindex(columns=fitur, fill_value=0)

    pred = int(model.predict(X_input)[0])
    proba = float(model.predict_proba(X_input)[0][1])

    st.divider()
    if pred == 1:
        st.error(f"⚠️ Mesin diprediksi **BERISIKO GAGAL** (probabilitas {proba:.1%})")
        st.write("Disarankan melakukan pemeriksaan atau perawatan mesin.")
    else:
        st.success(f"✅ Mesin diprediksi **AMAN** (probabilitas gagal {proba:.1%})")
        st.write("Mesin beroperasi dalam kondisi normal.")
    st.progress(min(max(proba, 0.0), 1.0), text="Probabilitas kegagalan")

    with st.expander("Lihat data yang dimasukkan"):
        st.dataframe(X_input)
