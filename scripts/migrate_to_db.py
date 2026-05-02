import json
import os
import sys
from sqlmodel import Session, select

# Add project root to path so we can import slopfinity
sys.path.append(os.getcwd())

from slopfinity.db import init_db, engine
from slopfinity.models import Configuration, QueueItem, ChatSession, ChatMessage, StoryLog
import slopfinity.config as cfg

def migrate():
    print("Initializing database...")
    init_db()
    
    with Session(engine) as session:
        # 1. Migrate Config
        if os.path.exists(cfg.CONFIG_FILE):
            print(f"Migrating config from {cfg.CONFIG_FILE}...")
            try:
                with open(cfg.CONFIG_FILE, "r") as f:
                    config_data = json.load(f)
                for key, value in config_data.items():
                    # Check if exists
                    existing = session.exec(select(Configuration).where(Configuration.key == key)).first()
                    if not existing:
                        session.add(Configuration(key=key, value=value))
                session.commit()
            except Exception as e:
                print(f"Error migrating config: {e}")
        
        # 2. Migrate Queue
        if os.path.exists(cfg.QUEUE_FILE):
            print(f"Migrating queue from {cfg.QUEUE_FILE}...")
            try:
                with open(cfg.QUEUE_FILE, "r") as f:
                    queue_data = json.load(f)
                for item in queue_data:
                    # Legacy migration logic from config.py/queue_schema.py
                    from slopfinity.queue_schema import migrate_legacy
                    item = migrate_legacy(item)
                    
                    # Check if exists
                    existing = session.get(QueueItem, item.get("id"))
                    if not existing:
                        q_item = QueueItem(
                            id=item.get("id"),
                            ts=item.get("ts"),
                            status=item.get("status", "pending"),
                            prompt=item.get("prompt"),
                            title=item.get("title"),
                            priority=item.get("priority", "normal"),
                            concurrent=bool(item.get("concurrent")),
                            infinity=bool(item.get("infinity")),
                            when_idle=bool(item.get("when_idle")),
                            chaos=bool(item.get("chaos")),
                            image_only=bool(item.get("image_only")),
                            fast_track=bool(item.get("fast_track")),
                            config_snapshot=item.get("config_snapshot", {}),
                            stages=item.get("stages", {}),
                            completed_ts=item.get("completed_ts"),
                            cancelled_ts=item.get("cancelled_ts"),
                            succeeded=item.get("succeeded"),
                            error=item.get("error"),
                            asset_paths=item.get("asset_paths", [])
                        )
                        session.add(q_item)
                session.commit()
            except Exception as e:
                print(f"Error migrating queue: {e}")

    print("Migration complete.")

if __name__ == "__main__":
    migrate()
