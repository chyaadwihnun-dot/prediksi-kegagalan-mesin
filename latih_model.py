"""Melatih model Random Forest (sama dengan Tugas Besar Part 3)
dan menyimpan model_rf.joblib + fitur.joblib di folder ini.
Juga mencetak hasil evaluasi untuk dipakai di slide/README."""
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

print("Mengunduh dataset AI4I 2020 (butuh internet)...")
df = fetch_ucirepo(id=601).data.original.copy()

rename = {}
for c in df.columns:
    l = c.lower()
    if l in ('uid', 'udi'): rename[c] = 'UDI'
    elif l.startswith('product'): rename[c] = 'Product ID'
    elif l.startswith('air'): rename[c] = 'Air temperature [K]'
    elif l.startswith('process'): rename[c] = 'Process temperature [K]'
    elif l.startswith('rotational'): rename[c] = 'Rotational speed [rpm]'
    elif l.startswith('torque'): rename[c] = 'Torque [Nm]'
    elif l.startswith('tool'): rename[c] = 'Tool wear [min]'
    elif l.startswith('machine'): rename[c] = 'Machine failure'
df = df.rename(columns=rename)
df = df.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
df = pd.get_dummies(df, columns=['Type'], drop_first=True)
df.columns = [c.replace('[', '(').replace(']', ')') for c in df.columns]

X = df.drop(columns=['Machine failure'])
y = df['Machine failure']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

best_rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=5,
    class_weight='balanced', random_state=42, n_jobs=-1)
best_rf.fit(X_train, y_train)

joblib.dump(best_rf, 'model_rf.joblib')
joblib.dump(list(X.columns), 'fitur.joblib')
print("\nModel tersimpan: model_rf.joblib dan fitur.joblib")
print("Fitur:", list(X.columns))

pred = best_rf.predict(X_test)
print("\n=== HASIL EVALUASI (data testing) ===")
print("Accuracy :", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred), 4))
print("Recall   :", round(recall_score(y_test, pred), 4))
print("F1-score :", round(f1_score(y_test, pred), 4))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification report:\n", classification_report(y_test, pred))
