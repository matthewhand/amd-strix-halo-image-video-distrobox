from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlmodel import SQLModel, Field, Column, JSON, create_engine, Session, select
import uuid

class Configuration(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: Any = Field(sa_column=Column(JSON))
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class QueueItem(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    ts: float = Field(index=True)
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
    
    # Tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_ts: Optional[float] = None
    cancelled_ts: Optional[float] = None
    succeeded: Optional[bool] = None
    error: Optional[str] = None
    asset_paths: List[str] = Field(default_factory=list, sa_column=Column(JSON))

class ChatSession(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True, index=True)

class ChatMessage(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    session_id: str = Field(foreign_key="chatsession.id", index=True)
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))
    ts: float = Field(default_factory=lambda: datetime.utcnow().timestamp())

class StoryLog(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    lines: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    active_idx: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True, index=True)
