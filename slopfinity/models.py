from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlmodel import SQLModel, Field, Column, JSON
import time
import uuid


def _utcnow() -> datetime:
    """Naive UTC now — non-deprecated replacement for datetime.utcnow().
    Kept tz-naive (offset stripped) to preserve the exact stored format."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

class Configuration(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: Any = Field(sa_column=Column(JSON))
    updated_at: datetime = Field(default_factory=_utcnow)

class QueueItem(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    # default to "now" so creating an item without an explicit ts (tests,
    # migrate_legacy) doesn't hit a NOT NULL IntegrityError on save.
    ts: float = Field(default_factory=time.time, index=True)
    # Persist the queue-schema version so a migrated (v2) item isn't re-migrated
    # on every read (which would reset its `stages`). default=1 ⇒ legacy items
    # without it get migrated once. See _migrate_sqlite_columns in db.py.
    schema_version: int = Field(default=1)
    status: str = Field(default="pending", index=True)
    prompt: str
    title: Optional[str] = None
    priority: str = Field(default="normal")
    concurrent: bool = Field(default=False)
    infinity: bool = Field(default=False)
    when_idle: bool = Field(default=False)
    chaos: bool = Field(default=False)
    image_only: bool = Field(default=False)
    fast_track: bool = Field(default=False)
    config_snapshot: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    stages: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    # Catch-all for queue fields that don't have a dedicated column yet — e.g.
    # seed_image, seed_images, seeds_mode, seed_prompt, stage_prompts,
    # stage_prompts_raw, polymorphic, started_ts, requeued_from_ts. Writers
    # (the /inject + run_fleet seed/FLF2V paths) stamp these onto the task dict;
    # without this column save_queue's `k in model_fields` filter would silently
    # drop them on the DB roundtrip, breaking those features. config.save_queue
    # funnels unknown keys in here and get_queue flattens them back to the top
    # level, so callers see a flat dict exactly as before.
    extra: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Tracking
    created_at: datetime = Field(default_factory=_utcnow)
    completed_ts: Optional[float] = None
    cancelled_ts: Optional[float] = None
    succeeded: Optional[bool] = None
    error: Optional[str] = None
    asset_paths: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    # Shared UUID that groups all queue items belonging to the same story batch.
    # None for standalone (non-story) injections. Set by the client via /inject.
    story_id: Optional[str] = None
    story_title: Optional[str] = None

class ChatSession(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    is_active: bool = Field(default=True, index=True)

class ChatMessage(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    session_id: str = Field(foreign_key="chatsession.id", index=True)
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))
    ts: float = Field(default_factory=lambda: _utcnow().timestamp())

class StoryLog(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    lines: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    active_idx: int = Field(default=0)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    is_active: bool = Field(default=True, index=True)
