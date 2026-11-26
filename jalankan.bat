@echo off
cd /d "%~dp0"
echo === Memasang library ===
python -m pip install -r requirements.txt
if not exist model_rf.joblib (
    echo === Model belum ada, melatih model... ===
    python latih_model.py
)
if not exist fitur.joblib (
    echo === fitur.joblib belum ada, melatih model... ===
    python latih_model.py
)
echo === Menjalankan aplikasi ===
python -m streamlit run app.py
pause
