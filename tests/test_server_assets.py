"""Route tests for the /assets and /asset/* endpoints."""
import os
import pytest
import pytest_asyncio
import unittest.mock as mock
from tests.conftest_server import client, default_config  # noqa: F401


pytestmark = pytest.mark.asyncio


class TestAssetsEndpoint:
    async def test_returns_empty_list_for_empty_dir(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("slopfinity.routers.assets.EXP_DIR", str(tmp_path))
        resp = await client.get("/assets")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    async def test_returns_asset_entries(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("slopfinity.routers.assets.EXP_DIR", str(tmp_path))
        (tmp_path / "FINAL_1.mp4").write_bytes(b"fake")
        (tmp_path / "frame.png").write_bytes(b"fake")
        resp = await client.get("/assets")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        names = {i["file"] for i in data["items"]}
        assert "FINAL_1.mp4" in names
        assert "frame.png" in names

    async def test_kind_classification(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("slopfinity.routers.assets.EXP_DIR", str(tmp_path))
        (tmp_path / "clip.mp4").write_bytes(b"x")
        (tmp_path / "img.png").write_bytes(b"x")
        resp = await client.get("/assets")
        items = {i["file"]: i["kind"] for i in resp.json()["items"]}
        assert items["clip.mp4"] == "video"
        assert items["img.png"] == "image"

    async def test_pagination_limit(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("slopfinity.routers.assets.EXP_DIR", str(tmp_path))
        for i in range(10):
            (tmp_path / f"clip_{i}.mp4").write_bytes(b"x")
        resp = await client.get("/assets?limit=3")
        data = resp.json()
        assert len(data["items"]) == 3
        assert data["total"] == 10
        assert data["has_more"] is True

    async def test_pagination_offset(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("slopfinity.routers.assets.EXP_DIR", str(tmp_path))
        for i in range(5):
            (tmp_path / f"clip_{i}.mp4").write_bytes(b"x")
        resp = await client.get("/assets?offset=3&limit=10")
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["has_more"] is False

    async def test_non_asset_files_excluded(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("slopfinity.routers.assets.EXP_DIR", str(tmp_path))
        (tmp_path / "notes.txt").write_bytes(b"x")
        (tmp_path / "clip.mp4").write_bytes(b"x")
        resp = await client.get("/assets")
        names = {i["file"] for i in resp.json()["items"]}
        assert "notes.txt" not in names
        assert "clip.mp4" in names


class TestAssetByVidx:
    async def test_returns_base_png(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("slopfinity.routers.assets.EXP_DIR", str(tmp_path))
        (tmp_path / "slop_1_dragons_base.png").write_bytes(b"x")
        resp = await client.get("/assets/by-vidx?v_idx=1")
        assert resp.status_code == 200
        assert "base" in resp.json()["assets"]

    async def test_returns_final_mp4(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("slopfinity.routers.assets.EXP_DIR", str(tmp_path))
        (tmp_path / "FINAL_2.mp4").write_bytes(b"x")
        resp = await client.get("/assets/by-vidx?v_idx=2")
        assert "final" in resp.json()["assets"]

    async def test_returns_empty_for_unknown_vidx(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("slopfinity.routers.assets.EXP_DIR", str(tmp_path))
        resp = await client.get("/assets/by-vidx?v_idx=999")
        assert resp.json()["assets"] == {}

    async def test_legacy_prefix_matched(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("slopfinity.routers.assets.EXP_DIR", str(tmp_path))
        (tmp_path / "v3_legacy_base.png").write_bytes(b"x")
        resp = await client.get("/assets/by-vidx?v_idx=3")
        assert "base" in resp.json()["assets"]


class TestAssetFileServing:
    async def test_existing_file_returned(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("slopfinity.routers.assets.EXP_DIR", str(tmp_path))
        f = tmp_path / "test.mp4"
        f.write_bytes(b"fakevideo")
        resp = await client.get("/asset/test.mp4")
        assert resp.status_code == 200

    async def test_missing_file_returns_404(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("slopfinity.routers.assets.EXP_DIR", str(tmp_path))
        resp = await client.get("/asset/nonexistent.mp4")
        assert resp.status_code == 404

    async def test_path_traversal_rejected(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("slopfinity.routers.assets.EXP_DIR", str(tmp_path))
        resp = await client.get("/asset/../etc/passwd")
        # Should be 404 (or 400) — never 200
        assert resp.status_code in (400, 403, 404, 422)
