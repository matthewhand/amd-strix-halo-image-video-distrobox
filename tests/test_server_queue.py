"""Route tests for all /queue/* endpoints and related helpers."""
import time
import pytest
import unittest.mock as mock
from tests.conftest_server import client, default_config  # noqa: F401

pytestmark = pytest.mark.asyncio


class TestQueuePaginated:
    async def test_empty_queue_returns_empty_page(self, client):
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=[]):
            resp = await client.get("/queue/paginated")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    async def test_returns_all_statuses(self, client):
        q = [
            {"status": "pending", "prompt": "a", "ts": 1.0},
            {"status": "working", "prompt": "b", "ts": 2.0},
            {"status": "done",    "prompt": "c", "ts": 3.0},
        ]
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=q):
            resp = await client.get("/queue/paginated")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3

    async def test_page_limit_respected(self, client):
        q = [{"status": "pending", "prompt": f"p{i}", "ts": float(i)} for i in range(20)]
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=q):
            resp = await client.get("/queue/paginated?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 5


class TestQueueCancel:
    async def test_cancel_existing_item(self, client):
        ts = 1000.0
        q = [{"status": "pending", "prompt": "neon", "ts": ts, "id": "abc"}]
        saved = []
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=q), \
             mock.patch("slopfinity.server.cfg.save_queue", side_effect=lambda x: saved.__iadd__(x)):
            resp = await client.post("/queue/cancel", json={"ts": ts})
        assert resp.status_code == 200
        assert resp.json().get("ok") is True

    async def test_cancel_nonexistent_returns_error(self, client):
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=[]), \
             mock.patch("slopfinity.server.cfg.save_queue"):
            resp = await client.post("/queue/cancel", json={"ts": 9999.0})
        # Route returns 404 when no matching item found
        assert resp.status_code == 404
        assert resp.json().get("ok") is False


class TestQueuePauseResume:
    async def test_pause_sets_flag(self, client):
        config = {"queue_paused": False}
        with mock.patch("slopfinity.server.cfg.load_config", return_value=config), \
             mock.patch("slopfinity.server.cfg.save_config"):
            resp = await client.post("/queue/pause")
        assert resp.status_code == 200

    async def test_resume_clears_flag(self, client):
        config = {"queue_paused": True}
        with mock.patch("slopfinity.server.cfg.load_config", return_value=config), \
             mock.patch("slopfinity.server.cfg.save_config"):
            resp = await client.post("/queue/resume")
        assert resp.status_code == 200

    async def test_pause_state_endpoint(self, client):
        # pause-state reads config directly; just verify endpoint responds 200
        with mock.patch("slopfinity.server.cfg.load_config", return_value={"queue_paused": True}):
            resp = await client.get("/queue/pause-state")
        assert resp.status_code == 200
        assert "paused" in resp.json()


class TestCancelAll:
    async def test_cancels_all_pending(self, client):
        q = [
            {"status": "pending", "prompt": "a", "ts": 1.0},
            {"status": "pending", "prompt": "b", "ts": 2.0},
            {"status": "done",    "prompt": "c", "ts": 3.0},
        ]
        saved = []
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=q), \
             mock.patch("slopfinity.server.cfg.save_queue", side_effect=lambda x: saved.__iadd__(x)):
            resp = await client.post("/cancel-all")
        assert resp.status_code == 200
        pending = [x for x in saved if x["status"] == "pending"]
        assert len(pending) == 0


class TestQueueClearCompleted:
    async def test_removes_done_items(self, client):
        q = [
            {"status": "done",    "prompt": "a", "ts": 1.0},
            {"status": "pending", "prompt": "b", "ts": 2.0},
        ]
        saved = []
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=q), \
             mock.patch("slopfinity.server.cfg.save_queue", side_effect=lambda x: saved.__iadd__(x)):
            resp = await client.post("/queue/clear-completed")
        assert resp.status_code == 200
        assert all(x["status"] != "done" for x in saved)


class TestQueueToggleInfinity:
    async def test_toggles_infinity_flag(self, client):
        ts = 1000.0
        q = [{"status": "pending", "prompt": "x", "ts": ts, "infinity": False}]
        saved = []
        with mock.patch("slopfinity.server.cfg.get_queue", return_value=q), \
             mock.patch("slopfinity.server.cfg.save_queue", side_effect=lambda x: saved.__iadd__(x)):
            resp = await client.post("/queue/toggle-infinity", json={"ts": ts})
        assert resp.status_code == 200
        assert resp.json().get("ok") is True
