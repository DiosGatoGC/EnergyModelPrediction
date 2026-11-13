"""Utilidades compartidas para acceder a PostgreSQL local sin credenciales en código."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_project_env() -> None:
    """Carga el .env local si existe; las variables del entorno tienen prioridad."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "Falta python-dotenv. Instale las dependencias con: pip install -r requirements.txt"
        ) from exc
    load_dotenv(project_root() / ".env", override=False)


def database_config() -> dict[str, str | int]:
    load_project_env()
    required = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Faltan variables de PostgreSQL: "
            + ", ".join(missing)
            + ". Copie .env.example como .env y complete sus valores locales."
        )
    return {
        "host": os.environ["DB_HOST"],
        "port": int(os.environ["DB_PORT"]),
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "5")),
    }


def connect_db(*, row_factory=None):
    """Abre una conexión psycopg y entrega un error útil cuando PostgreSQL no responde."""
    config = database_config()
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Falta psycopg. Instale las dependencias con: pip install -r requirements.txt"
        ) from exc

    try:
        kwargs = config
        if row_factory is not None:
            kwargs["row_factory"] = row_factory
        return psycopg.connect(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "No se pudo conectar a PostgreSQL local. Verifique que el servidor esté iniciado "
            "y que las variables DB_* en .env sean correctas."
        ) from exc
