"""Consulta y verifica, sin modificar datos, la corrida oficial Hito 3 en PostgreSQL local."""

from __future__ import annotations

import argparse

from db_utils import connect_db


def choose_run(cur, requested_run_id: str | None) -> dict:
    if requested_run_id:
        cur.execute("SELECT * FROM model_runs WHERE run_id = %s", (requested_run_id,))
    else:
        cur.execute("SELECT * FROM model_runs ORDER BY created_at DESC LIMIT 1")
    run = cur.fetchone()
    if run is None:
        raise RuntimeError("No hay corridas almacenadas. Ejecute scripts/persist_hito3_results.py primero.")
    return run


def main(run_id: str | None) -> None:
    try:
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "Falta psycopg. Instale las dependencias con: pip install -r requirements.txt"
        ) from exc
    with connect_db(row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database() AS database, current_user AS user")
            connection = cur.fetchone()
            cur.execute(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_schema = 'public'
                   AND table_name IN ('model_runs', 'predictions', 'segment_metrics')
                   ORDER BY table_name"""
            )
            tables = [row["table_name"] for row in cur.fetchall()]
            cur.execute("SELECT COUNT(*) AS total FROM model_runs")
            total_runs = cur.fetchone()["total"]
            run = choose_run(cur, run_id)

            cur.execute("SELECT COUNT(*) AS total FROM predictions WHERE run_id = %s", (run["run_id"],))
            prediction_count = cur.fetchone()["total"]
            cur.execute("SELECT COUNT(*) AS total FROM segment_metrics WHERE run_id = %s", (run["run_id"],))
            segment_count = cur.fetchone()["total"]

            cur.execute(
                """SELECT distribuidora, ubigeo, tarifa, target_periodo, target_real_mwh, prediction_mwh,
                          absolute_error_mwh
                   FROM predictions WHERE run_id = %s
                   ORDER BY absolute_error_mwh DESC LIMIT 5""",
                (run["run_id"],),
            )
            worst = cur.fetchall()
            cur.execute(
                """SELECT target_periodo, target_real_mwh, prediction_mwh, absolute_error_mwh
                   FROM predictions WHERE run_id = %s AND distribuidora = %s
                   ORDER BY target_periodo, ubigeo LIMIT 5""",
                (run["run_id"], "ENOSA"),
            )
            enosa_examples = cur.fetchall()
            cur.execute(
                """SELECT target_periodo, distribuidora, ubigeo, target_real_mwh, prediction_mwh
                   FROM predictions WHERE run_id = %s AND tarifa = %s
                   ORDER BY target_periodo, distribuidora, ubigeo LIMIT 5""",
                (run["run_id"], "BT5B"),
            )
            tariff_examples = cur.fetchall()

    print(f"Conexión correcta: base={connection['database']}, usuario={connection['user']}")
    print("Tablas encontradas:", ", ".join(tables))
    print(f"Corridas almacenadas: {total_runs}")
    print(f"Run verificado: {run['run_id']}")
    print(f"Modelo campeón: {run['model_name']}")
    print(f"Predicciones TEST: {prediction_count}")
    print(f"Métricas segmentadas: {segment_count}")
    print(
        "Métricas globales almacenadas: "
        f"MAE={run['test_mae']:.6f}, RMSE={run['test_rmse']:.6f}, "
        f"R2={run['test_r2']:.6f}, WAPE={run['test_wape']:.6f}"
    )
    print("Top 5 peores errores:")
    for row in worst:
        print(" ", row)
    print("Ejemplos ENOSA:")
    for row in enosa_examples:
        print(" ", row)
    print("Ejemplos tarifa BT5B:")
    for row in tariff_examples:
        print(" ", row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Verifica una corrida específica; por defecto, la más reciente.")
    args = parser.parse_args()
    try:
        main(args.run_id)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
