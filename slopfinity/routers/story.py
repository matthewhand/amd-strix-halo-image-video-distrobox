from fastapi import APIRouter, Body
from sqlmodel import Session, select
from slopfinity.db import engine
from slopfinity.models import StoryLog
from datetime import datetime, timezone

router = APIRouter()

@router.get("/story/log")
async def get_story_log():
    """Retrieve the active story log from the database."""
    with Session(engine) as session:
        active = session.exec(
            select(StoryLog).where(StoryLog.is_active == True)
        ).first()
        if not active:
            return {"ok": True, "lines": [], "active_idx": 0}
        return {
            "ok": True, 
            "lines": active.lines,
            "active_idx": active.active_idx,
            "id": active.id
        }

@router.post("/story/log")
async def update_story_log(payload: dict = Body(...)):
    """Wholesale replace the active story log (sync from client)."""
    lines = payload.get("lines") or []
    active_idx = payload.get("active_idx") or 0
    with Session(engine) as session:
        active = session.exec(
            select(StoryLog).where(StoryLog.is_active == True)
        ).first()
        if not active:
            active = StoryLog()
            session.add(active)
            session.commit()
            session.refresh(active)
        
        active.lines = lines
        active.active_idx = active_idx
        active.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        session.add(active)
        session.commit()
        return {"ok": True}

@router.delete("/story/log")
async def reset_story_log():
    """Archive the active story log and start a fresh one."""
    with Session(engine) as session:
        active = session.exec(
            select(StoryLog).where(StoryLog.is_active == True)
        ).first()
        if active:
            active.is_active = False
            active.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.add(active)
        
        new_log = StoryLog()
        session.add(new_log)
        session.commit()
        return {"ok": True, "id": new_log.id}
