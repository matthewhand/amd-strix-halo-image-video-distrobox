#!/usr/bin/env python3
"""
ComfyUI API client — submit workflows and monitor execution.

Usage:
    python comfyui_api.py workflow_api.json              # submit and monitor
    python comfyui_api.py workflow_api.json --no-wait     # fire and forget
    python comfyui_api.py workflow_api.json --batch 3     # submit N times with random seeds
"""
import argparse
import json
import random
import sys
import urllib.request

try:
    import websocket
    HAS_WS = True
except ImportError:
    HAS_WS = False

DEFAULT_SERVER = "127.0.0.1:8188"


def queue_prompt(workflow, server=DEFAULT_SERVER, client_id=None):
    payload = {"prompt": workflow}
    if client_id:
        payload["client_id"] = client_id
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"http://{server}/prompt", data=data)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def monitor_ws(server, client_id, prompt_id):
    """Watch execution progress via websocket."""
    if not HAS_WS:
        print("  (install 'websocket-client' for live progress)")
        return
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server}/ws?clientId={client_id}")
    try:
        while True:
            msg = ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                if data["type"] == "executing":
                    node = data["data"].get("node")
                    if node is None and data["data"]["prompt_id"] == prompt_id:
                        print("  Done.")
                        break
                    elif data["data"]["prompt_id"] == prompt_id:
                        print(f"  Executing node: {node}")
    finally:
        ws.close()


def main():
    parser = argparse.ArgumentParser(description="Submit workflows to ComfyUI API")
    parser.add_argument("workflow", help="Workflow JSON file (API format)")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="ComfyUI address")
    parser.add_argument("--no-wait", action="store_true", help="Don't monitor execution")
    parser.add_argument("--batch", type=int, default=1, help="Submit N copies with random seeds")
    parser.add_argument("--seed-node", default="8", help="Node ID containing noise_seed")
    args = parser.parse_args()

    with open(args.workflow) as f:
        data = json.load(f)
    workflow = data.get("prompt", data)

    import uuid
    client_id = str(uuid.uuid4())

    for i in range(args.batch):
        wf = json.loads(json.dumps(workflow))

        # Randomize seed
        if args.seed_node in wf and "inputs" in wf[args.seed_node]:
            seed = random.randint(1, 10**12)
            wf[args.seed_node]["inputs"]["noise_seed"] = seed

        label = f"[{i+1}/{args.batch}]" if args.batch > 1 else ""
        resp = queue_prompt(wf, args.server, client_id)
        prompt_id = resp.get("prompt_id", "unknown")
        print(f"{label} Queued: {prompt_id}")

        if not args.no_wait:
            monitor_ws(args.server, client_id, prompt_id)


if __name__ == "__main__":
    main()
