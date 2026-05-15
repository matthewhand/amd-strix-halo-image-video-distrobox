import os
import logging
from sqlmodel import create_engine, SQLModel, Session
from . import models

logger = logging.getLogger(__name__)

# Use SLOPFINITY_STATE_DIR or fallback to legacy relative path
_STATE_DIR = os.environ.get("SLOPFINITY_STATE_DIR") or "comfy-outputs/experiments"
DB_FILE = os.path.join(_STATE_DIR, "slopfinity.db")
sqlite_url = f"sqlite:///{DB_FILE}"

# check_same_thread=False is needed for SQLite with FastAPI/asyncio
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})


def _migrate_sqlite_columns():
    """Add any new columns to existing tables that SQLite won't auto-create.

    SQLite's CREATE TABLE IF NOT EXISTS won't add columns to tables that
    already exist. We use ALTER TABLE ... ADD COLUMN for each new field,
    ignoring the OperationalError that SQLite raises if the column already
    exists ('duplicate column name' is not a true error here).
    """
    new_columns = [
        # (table_name, column_name, column_def)
        ("queueitem", "story_id",    "TEXT"),
        ("queueitem", "story_title", "TEXT"),
    ]
    try:
        with engine.connect() as conn:
            for table, col, col_def in new_columns:
                try:
                    conn.execute(
                        __import__("sqlalchemy").text(
                            f'ALTER TABLE "{table}" ADD COLUMN "{col}" {col_def}'
                        )
                    )
                    conn.commit()
                    logger.info("db migration: added %s.%s", table, col)
                except Exception:
                    # Column already exists — expected on re-runs.
                    pass
    except Exception as exc:
        logger.warning("db migration skipped: %s", exc)


def init_db():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    SQLModel.metadata.create_all(engine)
    _migrate_sqlite_columns()


def get_session():
    with Session(engine) as session:
        yield session

