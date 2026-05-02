import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path

REPO_ROOT = Path(os.getcwd())
state_dir = Path(tempfile.mkdtemp(prefix="slopfinity-debug-"))
(state_dir / "tts").mkdir(exist_ok=True)
config = {
    "llm": {
        "provider": "lmstudio",
        "base_url": "http://127.0.0.1:1234/v1",
        "model_id": "mock-llm",
    },
}
(state_dir / "config.json").write_text(json.dumps(config))

app_env = os.environ.copy()
app_env["SLOPFINITY_STATE_DIR"] = str(state_dir)
app_env["PYTHONPATH"] = os.pathsep.join(sys.path) + os.pathsep + app_env.get("PYTHONPATH", "")
app_env["SLOPFINITY_TEST_MODE"] = "1"

print(f"Starting uvicorn in {state_dir}...")
subprocess.run([
    sys.executable, "-m", "uvicorn",
    "slopfinity.server:app",
    "--host", "127.0.0.1",
    "--port", "9101",
], env=app_env, cwd=str(REPO_ROOT))
