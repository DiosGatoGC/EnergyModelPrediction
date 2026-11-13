"""Persiste la corrida oficial de Hito 3 en PostgreSQL local, sin reentrenar modelos."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from db_utils import connect_db, project_root


ROOT = project_root()
OUTPUT_DIR = ROOT / "reports" / "model_outputs"
MANIFEST_PATH = OUTPUT_DIR / "hito3_run_manifest.json"
PREDICTIONS_PATH = OUTPUT_DIR / "predicciones_test_comparativa.csv"
TEST_METRICS_PATH = OUTPUT_DIR / "metricas_hito3_comparativa.csv"
VALIDATION_METRICS_PATH = OUTPUT_DIR / "metricas_hito3_validation.csv"

SEGMENT_SOURCES = (
    ("DISTRIBUIDORA", "hito3_errores_por_distribuidora.csv", ("DISTRIBUIDORA",)),
    ("TARIFA", "hito3_errores_por_tarifa.csv", ("TARIFA",)),
    ("DISTRIBUIDORA_UBIGEO", "hito3_errores_por_ubigeo.csv", ("DISTRIBUIDORA", "UBIGEO")),
    ("TARGET_PERIODO", "hito3_errores_por_mes.csv", ("TARGET_PERIODO",)),
    (
        "RANGO_CONSUMO",
        "hito3_errores_por_rango_consumo.csv",
        ("RANGO_CONSUMO_REAL", "LIMITE_INFERIOR_MWh", "LIMITE_SUPERIOR_MWh"),
    ),
)


def native(value: Any) -> Any:
    """Convierte valores pandas/numpy a tipos serializables por JSON y psycopg."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native(item) for item in value]
    return value


def number(value: Any) -> float | None:
    value = native(value)
    return None if value is None else float(value)


def load_official_artifacts() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (MANIFEST_PATH, PREDICTIONS_PATH, TEST_METRICS_PATH, VALIDATION_METRICS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró el artefacto oficial requerido: {path}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        dtype={"DISTRIBUIDORA": "string", "UBIGEO": "string", "TARIFA": "string"},
    )
    test_metrics = pd.read_csv(TEST_METRICS_PATH)
    validation_metrics = pd.read_csv(VALIDATION_METRICS_PATH)

    expected_prediction_columns = {
        "DISTRIBUIDORA", "UBIGEO", "TARIFA", "PERIODO", "TARGET_PERIODO",
        "TARGET_REAL", "PRED_CAMPEON", "RESIDUO", "ERROR_ABSOLUTO",
    }
    missing = expected_prediction_columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"El archivo oficial de predicciones no contiene: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("El archivo oficial de predicciones TEST está vacío.")
    if not predictions["TARGET_PERIODO"].astype(int).between(202504, 202507).all():
        raise ValueError("Las predicciones seleccionadas no corresponden exclusivamente a TEST oficial.")
    return manifest, predictions, test_metrics, validation_metrics


def stable_run_id(manifest: dict[str, Any]) -> str:
    identity = {
        "run_name": manifest["run_name"],
        "executed_at": manifest["executed_at"],
        "dataset": manifest["dataset"],
        "target": manifest["target"],
        "champion": manifest["champion"],
        "selection_metric": manifest["selection_metric"],
    }
    raw = json.dumps(native(identity), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def split_periods(manifest: dict[str, Any]) -> dict[str, int]:
    values = {name: [int(item) for item in re.findall(r"\d{6}", expression)] for name, expression in manifest["splits"].items()}
    if len(values["TRAIN"]) != 1 or len(values["VALIDATION"]) != 2 or len(values["TEST"]) != 2:
        raise ValueError("El manifiesto no contiene los límites temporales esperados.")
    return {
        "train_end": values["TRAIN"][0],
        "validation_start": values["VALIDATION"][0],
        "validation_end": values["VALIDATION"][1],
        "test_start": values["TEST"][0],
        "test_end": values["TEST"][1],
    }


def champion_metrics(frame: pd.DataFrame, champion_name: str) -> dict[str, Any]:
    match = frame.loc[frame["MODELO"] == champion_name]
    if len(match) != 1:
        raise ValueError(f"No se encontró una única fila de métricas para el campeón: {champion_name}")
    return native(match.iloc[0].to_dict())


def segment_records(run_id: str) -> list[tuple[Any, ...]]:
    records: list[tuple[Any, ...]] = []
    for segment_type, filename, value_columns in SEGMENT_SOURCES:
        path = OUTPUT_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Falta el output oficial de segmento: {path}")
        frame = pd.read_csv(path, dtype={"UBIGEO": "string", "DISTRIBUIDORA": "string", "TARIFA": "string"})
        required = {"OBSERVACIONES", "MAE_MWh", "RMSE_MWh", "R2", "WAPE_pct", "RESIDUO_MEDIO", *value_columns}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{filename} no contiene columnas requeridas: {sorted(missing)}")
        for row in frame.to_dict(orient="records"):
            segment_value = {column: native(row[column]) for column in value_columns}
            records.append((
                run_id,
                segment_type,
                segment_value,
                int(row["OBSERVACIONES"]),
                number(row["MAE_MWh"]),
                number(row["RMSE_MWh"]),
                number(row["R2"]),
                number(row["WAPE_pct"]),
                number(row["RESIDUO_MEDIO"]),
            ))
    return records


def close_enough(actual: float | None, expected: float | None, label: str) -> None:
    if actual is None and expected is None:
        return
    if actual is None or expected is None or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"Diferencia en {label}: BD={actual}, archivo={expected}")


def verify_inserted_run(cur, run_id: str, predictions: pd.DataFrame, validation: dict[str, Any], test: dict[str, Any], segments: list[tuple[Any, ...]]) -> None:
    cur.execute("SELECT COUNT(*) FROM predictions WHERE run_id = %s", (run_id,))
    stored_predictions = cur.fetchone()[0]
    if stored_predictions != len(predictions):
        raise AssertionError(
            f"Conteo inconsistente: archivo oficial={len(predictions)}, PostgreSQL={stored_predictions}."
        )

    cur.execute(
        """SELECT validation_mae, validation_rmse, validation_r2, validation_wape,
                  test_mae, test_rmse, test_r2, test_wape
           FROM model_runs WHERE run_id = %s""",
        (run_id,),
    )
    stored = cur.fetchone()
    expected = (
        validation["MAE_MWh"], validation["RMSE_MWh"], validation["R2"], validation["WAPE_pct"],
        test["MAE_MWh"], test["RMSE_MWh"], test["R2"], test["WAPE_pct"],
    )
    for label, actual, source in zip(
        ("validation MAE", "validation RMSE", "validation R2", "validation WAPE", "test MAE", "test RMSE", "test R2", "test WAPE"),
        stored,
        expected,
    ):
        close_enough(float(actual), number(source), label)

    cur.execute(
        """SELECT segment_type, segment_value, n_observations, mae, rmse, r2, wape, mean_residual
           FROM segment_metrics WHERE run_id = %s""",
        (run_id,),
    )
    stored_segments = cur.fetchall()
    if len(stored_segments) != len(segments):
        raise AssertionError(f"Segmentos inconsistentes: archivo={len(segments)}, PostgreSQL={len(stored_segments)}.")

    def key(record: tuple[Any, ...]) -> tuple[str, str]:
        return record[1], json.dumps(native(record[2]), sort_keys=True, separators=(",", ":"))

    expected_by_key = {key(record): record for record in segments}
    for row in stored_segments:
        actual_key = (row[0], json.dumps(native(row[1]), sort_keys=True, separators=(",", ":")))
        expected_row = expected_by_key.get(actual_key)
        if expected_row is None:
            raise AssertionError(f"Segmento inesperado en PostgreSQL: {actual_key}")
        if int(row[2]) != expected_row[3]:
            raise AssertionError(f"Observaciones inconsistentes para segmento {actual_key}")
        for label, actual, source in zip(("MAE", "RMSE", "R2", "WAPE", "residuo medio"), row[3:], expected_row[4:]):
            close_enough(number(actual), number(source), f"{label} de {actual_key}")


def persist(replace: bool) -> str:
    try:
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise RuntimeError(
            "Falta psycopg. Instale las dependencias con: pip install -r requirements.txt"
        ) from exc

    manifest, predictions, test_metrics, validation_metrics = load_official_artifacts()
    champion = manifest["champion"]["name"]
    run_id = stable_run_id(manifest)
    periods = split_periods(manifest)
    validation = champion_metrics(validation_metrics, champion)
    test = champion_metrics(test_metrics, champion)
    segments = segment_records(run_id)

    prediction_rows = [
        (
            run_id,
            str(row["DISTRIBUIDORA"]), str(row["UBIGEO"]), str(row["TARIFA"]),
            int(row["PERIODO"]), int(row["TARGET_PERIODO"]), number(row["TARGET_REAL"]),
            number(row["PRED_CAMPEON"]), number(row["RESIDUO"]), number(row["ERROR_ABSOLUTO"]),
            number(row.get("ERROR_PORCENTUAL")),
        )
        for row in predictions.to_dict(orient="records")
    ]

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM model_runs WHERE run_id = %s", (run_id,))
            if cur.fetchone():
                if not replace:
                    raise RuntimeError(
                        f"La corrida {run_id} ya existe. No se insertó nada. Use --replace para sustituirla explícitamente."
                    )
                cur.execute("DELETE FROM model_runs WHERE run_id = %s", (run_id,))

            cur.execute(
                """INSERT INTO model_runs (
                    run_id, created_at, run_name, model_name, model_version, dataset_name, target_name,
                    selection_metric, validation_mae, validation_rmse, validation_r2, validation_wape,
                    test_mae, test_rmse, test_r2, test_wape, train_end_period, validation_start_period,
                    validation_end_period, test_start_period, test_end_period, random_state, cv_strategy,
                    hyperparameters, features, manifest
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    run_id, datetime.fromisoformat(manifest["executed_at"]), manifest["run_name"], champion,
                    manifest["run_name"], manifest["dataset"], manifest["target"], manifest["selection_metric"],
                    number(validation["MAE_MWh"]), number(validation["RMSE_MWh"]), number(validation["R2"]), number(validation["WAPE_pct"]),
                    number(test["MAE_MWh"]), number(test["RMSE_MWh"]), number(test["R2"]), number(test["WAPE_pct"]),
                    periods["train_end"], periods["validation_start"], periods["validation_end"], periods["test_start"], periods["test_end"],
                    int(manifest["random_state"]), json.dumps(native(manifest["cv"]), ensure_ascii=False),
                    Jsonb(native(manifest["champion"]["hyperparameters"])), Jsonb(native(manifest["features"])), Jsonb(native(manifest)),
                ),
            )
            cur.executemany(
                """INSERT INTO predictions (
                    run_id, distribuidora, ubigeo, tarifa, periodo, target_periodo, target_real_mwh,
                    prediction_mwh, residual_mwh, absolute_error_mwh, percentage_error
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                prediction_rows,
            )
            cur.executemany(
                """INSERT INTO segment_metrics (
                    run_id, segment_type, segment_value, n_observations, mae, rmse, r2, wape, mean_residual
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [(row[0], row[1], Jsonb(row[2]), *row[3:]) for row in segments],
            )
            verify_inserted_run(cur, run_id, predictions, validation, test, segments)

    print(f"Corrida persistida y verificada: {run_id}")
    print(f"Predicciones TEST: archivo={len(predictions)}, PostgreSQL={len(predictions)}")
    print(f"Métricas segmentadas almacenadas: {len(segments)}")
    print(
        "Métricas TEST del campeón: "
        f"MAE={test['MAE_MWh']:.6f}, RMSE={test['RMSE_MWh']:.6f}, R2={test['R2']:.6f}, WAPE={test['WAPE_pct']:.6f}"
    )
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="Reemplaza explícitamente una corrida con el mismo run_id.")
    args = parser.parse_args()
    try:
        persist(replace=args.replace)
    except (AssertionError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
