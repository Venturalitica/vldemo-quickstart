"""Stage `featurize` del quickstart — carga el dataset REAL de educación (UCI Student
Performance) y produce `data/features.parquet` con las columnas que el eval agnóstico
(`compliance_eval`) espera: `target` (aprobado G3>=10 → 1), `gender` (sex M→1/F→0) y `age`.
El resto de columnas son features. Stage estable frente al tratamiento (cambia train.py)."""

import pandas as pd

# El CSV de UCI usa separador ';'. Se commitea en data/ por el escenario/workflow.
df = pd.read_csv("data/student-mat.csv", sep=";")

# Etiqueta binaria: aprobado si la nota final G3 >= 10 (escala 0-20).
df["target"] = (df["G3"] >= 10).astype(int)
# Atributo protegido para la paridad demográfica: sex (M/F) → gender (1/0).
df["gender"] = (df["sex"].astype(str).str.upper() == "M").astype(int)
# `age` ya es numérica en el dataset.

# Quita las notas de periodo (G1/G2/G3) para que el modelo no sea trivial por fuga de la nota.
df = df.drop(columns=["G1", "G2", "G3", "sex"])

df.to_parquet("data/features.parquet")
print("featurize → data/features.parquet")
