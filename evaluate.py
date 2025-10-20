"""Stage `evaluate` del pipeline **DVC** (cat 1) — corre el eval AGNÓSTICO sobre el dato del
stage `featurize` y persiste el modelo como out cacheado.

Concerns DVC aquí (y SOLO aquí, no en el core agnóstico): (1) leer `data/features.parquet` —la
arista real del DAG `featurize → evaluate`—; (2) volcar `model.pkl` como **out CACHEADO** (DVC
versiona el MODELO entrenado, Art.15, content-addressed). La medición (venturalitica-sdk) vive
en `compliance_eval`; el tratamiento (variante) en `train.py`.
"""

import joblib
import pandas as pd

import compliance_eval
import train

df = pd.read_parquet("data/features.parquet")  # out del stage `featurize` (arista del DAG)
_, model = compliance_eval.run(train.build_model, df)
joblib.dump(model, "model.pkl")  # out cacheado: DVC versiona el modelo (Art.15)
