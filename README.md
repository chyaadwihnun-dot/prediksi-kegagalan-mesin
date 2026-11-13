# Prediksi Kegagalan Mesin (Predictive Maintenance)

Aplikasi web (Gradio) untuk memprediksi kegagalan mesin memakai model Random Forest
yang dilatih dari dataset AI4I 2020 (Tugas Besar Machine Learning).

## Cara menjalankan
```
pip install -r requirements.txt
python app.py
```
Buka http://127.0.0.1:7860

## File
- `app.py` : aplikasi dan User Interface
- `model_rf.joblib` : model Random Forest hasil training di Google Colab
- `fitur.joblib` : urutan fitur model
