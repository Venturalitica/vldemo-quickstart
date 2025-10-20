"""Núcleo del eval del quickstart-education — **AGNÓSTICO al framework MLOps** (no importa DVC/MLflow/Dagster).

Aquí vive TODO lo que NO cambia entre escenarios ni entre tratamientos:
  · carga del dato FIRST-CLASS vía **Croissant** (§2, `mlcroissant` — no `read_csv`),
  · la **evaluación de cumplimiento** con el **venturalitica-sdk** (`vl.monitor` + `vl.enforce`
    en dos fases: Art.10 datos + Art.15 modelo) — VISIBLE aquí, es EL eval,
  · el volcado de `metrics.json` plano `{control_id: actual_value}`.

Lo que SÍ cambia se inyecta:
  · el **TRATAMIENTO** (variante de entrenamiento V1/V2) → `build_model`, vive en `train.py`;
  · lo que cada **framework** hace con el modelo (DVC lo cachea, MLflow lo registra, Dagster lo
    expone) → vive en el fichero de ese framework (`evaluate.py`/`mlflow_entry.py`/`dagster_defs.py`),
    NUNCA aquí. Por eso al mirar un escenario no se ve el código de otro backend.

NO juzga: el veredicto autoritativo lo pone el motor venth (Rust) contra el mismo OSCAL (§6.6).
"""

import os

os.environ.setdefault("VENTURALITICA_NO_ANALYTICS", "1")  # sin telemetría en CI

import contextlib
import json
import shutil
import sys
from pathlib import Path

import mlcroissant as mlc
import numpy as np
import pandas as pd
import yaml

import venturalitica as vl

# Rutas FIJAS relativas a la raíz del repo (= cwd del recómputo, cualquiera sea el backend).
CROISSANT = "data/student.croissant.json"
OSCAL = "shared_data/policies/assessment_plan.oscal.yaml"
PARAMS = "params.yaml"
METRICS = "metrics.json"
BOM_ROOT = ".venturalitica/bom.json"
RUNS_DIR = Path(".venturalitica/runs")

TARGET = "target"
GENDER = "gender"
AGE = "age"
AGE_BUCKET = "age_bucket"  # dimensión por GRUPO de edad (terciles) para la paridad demográfica
AGE_BUCKETS = 3


def add_age_buckets(cohort: pd.DataFrame) -> pd.DataFrame:
    """Materializa la columna derivada `age_bucket` (terciles de `age`) en la cohorte — concern
    de MEDICIÓN, agnóstico al tratamiento. La paridad demográfica por edad se mide por GRUPO de
    edad (la edad cruda tiene grupos unipersonales → demographic_parity_diff por año es
    inalcanzable); los measures que la usan declaran `dimension: age_bucket` + `age_buckets: 3`.
    Se respeta el bucket si el tratamiento ya lo fijó; es columna no usada por escenarios cuyos
    measures referencian `age` cruda → cambio inerte para ellos."""
    if AGE in cohort.columns and AGE_BUCKET not in cohort.columns:
        qs = [i / AGE_BUCKETS for i in range(1, AGE_BUCKETS)]
        edges = cohort[AGE].quantile(qs).to_numpy()
        cohort = cohort.copy()
        cohort[AGE_BUCKET] = cohort[AGE].map(
            lambda a: f"b{int(np.searchsorted(edges, a, side='right'))}"
        )
    return cohort


def load_applications(croissant_path: str = CROISSANT) -> pd.DataFrame:
    """Carga el dataset FIRST-CLASS vía Croissant (§2): el Croissant-RAI declara cómo leerlo
    (distribution + recordSet) y `mlcroissant` lo materializa — no leemos el CSV a mano.
    mlcroissant nombra los campos `applications/<col>` y devuelve texto en bytes."""
    ds = mlc.Dataset(jsonld=croissant_path)
    df = pd.DataFrame(list(ds.records(record_set="students")))
    df = df.rename(columns=lambda c: c.split("/", 1)[1] if "/" in c else c)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
    df[GENDER] = (df[GENDER].astype(str).str.strip().str.lower() == "male").astype(int)
    return df


def seed(params_path: str = PARAMS) -> int:
    """Semilla del modelo (determinismo del entrenamiento, Art.15) desde params.yaml."""
    return int(yaml.safe_load(open(params_path))["seed"])


def _control_order(oscal_path: str) -> dict:
    """Orden de los controles en el assessment_plan → orden canónico del dossier."""
    doc = yaml.safe_load(open(oscal_path))
    reqs = doc["component-definition"]["components"][0]["control-implementations"][0][
        "implemented-requirements"
    ]
    return {r["control-id"]: i for i, r in enumerate(reqs)}


def _metric_entry(result) -> float | dict:
    """Una entrada de `metrics.json` para un `ComplianceResult` (contrato §3.2, power-stats):

    - SDK ≥0.6.11 expone `result.power` (dict {n, ci_low, ci_high, …}, fiabilidad por bootstrap)
      → forma OBJETO `{value, power}` (evidencia reforzada: el IC del estimador entra en `venth.lock`);
    - SDK 0.6.10 NO tiene ese atributo → forma ESCALAR `value` (back-compat: el loan sigue verde hoy).

    El núcleo Rust acepta AMBAS formas (`MetricEntry` untagged); por eso el bump del SDK añade poder
    sin romper nada. NO juzgamos aquí: el veredicto autoritativo lo pone el motor venth vs el OSCAL."""
    value = float(result.actual_value)
    power = getattr(result, "power", None)
    if power:
        return {"value": value, "power": power}
    return value


def _enforce_phases(df: pd.DataFrame, cohort: pd.DataFrame, oscal_path: str) -> dict:
    """Las dos fases de `vl.enforce` (Art.10 datos / Art.15 modelo) sobre el assessment_plan →
    dict {control_id: entry} en el orden del OSCAL, donde `entry` es escalar (0.6.10) u objeto
    `{value, power}` (≥0.6.11). **DEBE ejecutarse DENTRO de un `vl.monitor()`** (lo abre `run`)
    para que el SDK ligue los resultados al modelo trazado."""
    data_results = vl.enforce(
        data=df, policy=oscal_path, target=TARGET, gender=GENDER, age=AGE,
        phase="training", strict=False,
    )
    model_results = vl.enforce(
        data=cohort, policy=oscal_path, target=TARGET, prediction="prediction",
        gender=GENDER, age=AGE, phase="validation", strict=False,
    )
    order = _control_order(oscal_path)
    results = sorted(data_results + model_results, key=lambda r: order.get(r.control_id, 10**6))
    return {r.control_id: _metric_entry(r) for r in results}


def _predict(model, X) -> np.ndarray:
    """Decisión binaria del modelo, agnóstica al tipo: las reducciones de fairlearn
    (`ExponentiatedGradient`) no exponen `predict`/`predict_proba` estándar → se usa `_pmf_predict`
    (la misma vía que el tratamiento), umbral 0.5; los estimadores sklearn usan `.predict`."""
    if hasattr(model, "_pmf_predict"):
        return (model._pmf_predict(X)[:, 1] >= 0.5).astype(int)
    return np.asarray(model.predict(X)).astype(int)


def _robustness_stability(model, X, seed_val: int) -> float:
    """Art. 15 robustez (ISO/IEC 24029-2): fracción de predicciones que NO cambian ante una
    perturbación PEQUEÑA y SEMBRADA (gaussiana, 1% de la desviación por feature) de la matriz de
    entrada del modelo. Determinista. El SDK no computa robustez → se mide aquí. La línea base se
    toma POR LA MISMA VÍA numérica que la perturbada (`_predict(model, Xv)`) para aislar SOLO el
    efecto de la perturbación (no una diferencia DataFrame↔ndarray del path de predicción)."""
    Xv = np.asarray(X, dtype=float)
    if Xv.size == 0:
        return 1.0
    rng = np.random.default_rng(seed_val)
    scale = 0.01 * (np.std(Xv, axis=0) + 1e-9)
    base = _predict(model, Xv)
    perturbed = _predict(model, Xv + rng.normal(0.0, 1.0, Xv.shape) * scale)
    return float(np.mean(base == perturbed))


def _artifact_integrity(build_model, df: pd.DataFrame, base_pred, seed_val: int) -> float:
    """Art. 15 ciber/integridad (ISO/IEC 27001 — integridad/reproducibilidad del artefacto):
    re-entrena con la MISMA semilla y comprueba que produce predicciones IDÉNTICAS → 1.0 si el
    artefacto es reproducible (determinista), <1.0 si hay no-determinismo. Es la dimensión ciber
    MEDIBLE de un modelo (la integridad criptográfica de la evidencia la da la firma DSSE)."""
    cohort2, _, _ = build_model(df, seed_val)
    p2 = np.asarray(cohort2["prediction"]).astype(int)
    bp = np.asarray(base_pred).astype(int)
    return 0.0 if len(bp) != len(p2) else float(np.mean(bp == p2))


def _promote_bom() -> None:
    """Promueve el bom.json que BOMProbe dejó en .venturalitica/runs/<run>/ a la raíz
    .venturalitica/bom.json (lo que read_bom lee, bom.rs:15). Elige el run con mtime máximo
    (misma heurística que el CLI push). FAIL-LOUD si no hay ningún bom.json que promover."""
    candidates = sorted(
        (p for p in RUNS_DIR.glob("*/bom.json") if p.parent.name != "latest"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit("compliance_eval: no se generó ningún bom.json en .venturalitica/runs/")
    Path(".venturalitica").mkdir(exist_ok=True)
    shutil.copyfile(candidates[-1], BOM_ROOT)
    print(f"bom → {BOM_ROOT} (desde {candidates[-1]})", file=sys.stderr)


def write_metrics(metrics: dict, path: str = METRICS) -> None:
    """Métricas `metrics.json` — contrato §3.2 que leen todos los backends: `{control_id: valor}`
    (0.6.10) o `{control_id: {value, power}}` (≥0.6.11). Misma forma para DVC/MLflow/Dagster."""
    json.dump(metrics, open(path, "w"), indent=2)


def run(build_model, df: pd.DataFrame | None = None, oscal_path: str = OSCAL):
    """Orquesta la evaluación AGNÓSTICA: entrena (variante = tratamiento) → mide (vl) → vuelca
    metrics.json. Devuelve `(cohort, model)` para que el backend haga lo suyo con el modelo
    entrenado (cachearlo como out, registrarlo, exponerlo como metadata). `df` se inyecta cuando
    el backend ya lo materializó (p.ej. el parquet del stage `featurize` de DVC); si no, se carga
    vía Croissant (§2).

    **`vl.monitor()` RECUBRE EL ENTRENAMIENTO Y LA VALIDACIÓN** (no solo el enforce): así el SDK
    captura las trazas del MODELO (sesión + BOM + linaje del `fit`) — imprescindible para alimentar
    el Anexo IV (§2 desarrollo del modelo). El ruido del SDK → stderr (stdout limpio p.ej. MLflow)."""
    if df is None:
        df = load_applications()
    with contextlib.redirect_stdout(sys.stderr):
        with vl.monitor(name="quickstart-education", label="venth eval"):
            cohort, model, x = build_model(df, seed())   # ENTRENAMIENTO (Art.15) — trazado por el SDK
            cohort = add_age_buckets(cohort)             # dimensión por grupo de edad (medición)
            metrics = _enforce_phases(df, cohort, oscal_path)  # validación (Art.10/Art.15)
            # Art. 15 medido aquí (el SDK no computa robustez/integridad): robustez ante perturbación
            # (ISO 24029) + reproducibilidad del artefacto (ISO 27001). Deterministas, advisory.
            base_pred = cohort["prediction"].to_numpy() if "prediction" in cohort.columns else _predict(model, x)
            metrics["robustness-stability"] = _robustness_stability(model, x, seed())
            metrics["artifact-integrity"] = _artifact_integrity(build_model, df, base_pred, seed())
        _promote_bom()  # tras cerrar la sesión, el bom.json del run ya existe
    write_metrics(metrics)
    return cohort, model
