"""Tratamiento del quickstart — modelo mínimo (LogisticRegression). Para el caso gratis NO hay
bucle rojo→verde (eso lo enseñan los ejemplos cargados); aquí basta un modelo razonable que dé
un veredicto ISO 23894 honesto. Descarta el atributo protegido `gender` de las features."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "target"
GENDER = "gender"
AGE = "age"


def build_model(df: pd.DataFrame, seed: int):
    """Devuelve (cohort con `prediction`, modelo sklearn, X). Descarta el protegido `gender`."""
    leaky = [TARGET, "prediction", GENDER]
    X = pd.get_dummies(df.drop(columns=[c for c in leaky if c in df.columns]))
    y = df[TARGET]
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = make_pipeline(
        StandardScaler(with_mean=False), LogisticRegression(max_iter=5000, random_state=seed)
    ).fit(Xtr, ytr)
    cohort = df.copy()
    cohort["prediction"] = model.predict(X)
    return cohort, model, X
