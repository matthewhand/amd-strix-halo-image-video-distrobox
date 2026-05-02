import os
from sqlmodel import create_engine, SQLModel, Session
from . import models

# Use SLOPFINITY_STATE_DIR or fallback to legacy relative path
_STATE_DIR = os.environ.get("SLOPFINITY_STATE_DIR") or "comfy-outputs/experiments"
DB_FILE = os.path.join(_STATE_DIR, "slopfinity.db")
sqlite_url = f"sqlite:///{DB_FILE}"

# check_same_thread=False is needed for SQLite with FastAPI/asyncio
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})

def init_db():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
