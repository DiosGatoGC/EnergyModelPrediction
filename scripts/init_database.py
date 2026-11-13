"""Inicializa de forma idempotente el esquema PostgreSQL local del proyecto."""

from __future__ import annotations

from pathlib import Path

from db_utils import connect_db, project_root


SCHEMA_PATH = project_root() / "database" / "schema.sql"


def main() -> None:
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"No se encontró el schema: {SCHEMA_PATH}")
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    try:
        with connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute(schema)
                cur.execute(
                    """SELECT table_name FROM information_schema.tables
                       WHERE table_schema = 'public'
                       AND table_name IN ('model_runs', 'predictions', 'segment_metrics')
                       ORDER BY table_name"""
                )
                tables = [row[0] for row in cur.fetchall()]
    except RuntimeError:
        raise
    print("Esquema inicializado sin borrar datos.")
    print("Tablas disponibles:", ", ".join(tables))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
