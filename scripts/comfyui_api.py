#!/usr/bin/env python3
"""
ComfyUI API client — full CLI for the HTTP/WebSocket API.

Subcommands:
    status          Health check (GET /system_stats)
    queue           Show running + pending jobs (GET /queue)
    nodes           List/search available node types (GET /object_info)
    schema NODE     Show a single node's input/output schema
    submit FILE     Submit API-format workflow JSON (POST /prompt)
    watch ID        Monitor a running job by prompt_id (websocket)
    history [ID]    List jobs or show one job's outputs
    clear           Clear pending queue (POST /queue clear)
    cancel ID       Cancel a specific job (POST /queue delete)
    interrupt       Stop current execution (POST /interrupt)

Examples:
    python comfyui_api.py status
    python comfyui_api.py nodes --grep LTXV
    python comfyui_api.py schema LTXVImgToVideoConditionOnly
    python comfyui_api.py submit workflow.json --watch
    python comfyui_api.py submit workflow.json --batch 3 --seed-node 18
    python comfyui_api.py queue
    python comfyui_api.py history abc123-def
"""
import argparse
import json
import random
import sys
import urllib.error
import urllib.request
import uuid

try:
    import websocket
    HAS_WS = True
except ImportError:
    HAS_WS = False

DEFAULT_SERVER = "127.0.0.1:8188"


# --- Low-level HTTP helpers ---

def _get(server, path):
    with urllib.request.urlopen(f"http://{server}{path}") as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(server, path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"http://{server}{path}", data=data)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


# --- Public API ---

def health(server):
    return _get(server, "/system_stats")


def list_nodes(server):
    return _get(server, "/object_info")


def node_schema(server, node_name):
    info = _get(server, "/object_info")
    return info.get(node_name)


def queue_info(server):
    return _get(server, "/queue")


def history(server, prompt_id=None):
    path = f"/history/{prompt_id}" if prompt_id else "/history"
    return _get(server, path)


def submit(workflow, server, client_id=None):
    payload = {"prompt": workflow}
    if client_id:
        payload["client_id"] = client_id
    return _post(server, "/prompt", payload)


def clear_queue(server):
    return _post(server, "/queue", {"clear": True})


def cancel_job(server, prompt_id):
    return _post(server, "/queue", {"delete": [prompt_id]})


def interrupt(server):
    return _post(server, "/interrupt", {})


def watch(server, client_id, prompt_id):
    """Block until a job completes, printing progress events."""
    if not HAS_WS:
        print("(install 'websocket-client' for live progress)")
        return
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server}/ws?clientId={client_id}")
    try:
        while True:
            msg = ws.recv()
            if not isinstance(msg, str):
                continue
            data = json.loads(msg)
            mtype = data.get("type")
            if mtype == "executing":
                dd = data["data"]
                if dd.get("prompt_id") != prompt_id:
                    continue
                if dd.get("node") is None:
                    print("Done.")
                    break
                print(f"  Executing node: {dd['node']}")
            elif mtype == "progress":
                dd = data["data"]
                if dd.get("prompt_id") == prompt_id:
                    print(f"  Step {dd.get('value')}/{dd.get('max')}")
    finally:
        ws.close()


# --- CLI ---

def cmd_status(args):
    data = health(args.server)
    sys_info = data.get("system", {})
    devs = data.get("devices", [])
    print(f"ComfyUI {sys_info.get('comfyui_version', '?')} on {sys_info.get('os', '?')}")
    print(f"Python {sys_info.get('python_version', '?')}  PyTorch {sys_info.get('pytorch_version', '?')}")
    for d in devs:
        vram_used = d.get("vram_total", 0) - d.get("vram_free", 0)
        print(f"  GPU: {d.get('name', '?')}  VRAM: {vram_used/1e9:.1f}/{d.get('vram_total',0)/1e9:.1f} GB")


def cmd_nodes(args):
    data = list_nodes(args.server)
    names = sorted(data.keys())
    if args.grep:
        names = [n for n in names if args.grep.lower() in n.lower()]
    for n in names:
        print(n)
    print(f"\n({len(names)} nodes)", file=sys.stderr)


def cmd_schema(args):
    data = node_schema(args.server, args.node)
    if data is None:
        print(f"Node '{args.node}' not found", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(data, indent=2))


def cmd_submit(args):
    with open(args.workflow) as f:
        data = json.load(f)
    workflow = data.get("prompt", data)

    client_id = str(uuid.uuid4())
    for i in range(args.batch):
        wf = json.loads(json.dumps(workflow))
        if args.batch > 1 and args.seed_node in wf and "inputs" in wf[args.seed_node]:
            seed = random.randint(1, 10**12)
            wf[args.seed_node]["inputs"]["noise_seed"] = seed

        label = f"[{i+1}/{args.batch}] " if args.batch > 1 else ""
        try:
            resp = submit(wf, args.server, client_id)
            pid = resp.get("prompt_id", "unknown")
            print(f"{label}Queued: {pid}")
            if args.watch:
                watch(args.server, client_id, pid)
        except RuntimeError as e:
            print(f"{label}FAIL: {e}", file=sys.stderr)
            sys.exit(1)


def cmd_queue(args):
    data = queue_info(args.server)
    running = data.get("queue_running", [])
    pending = data.get("queue_pending", [])
    print(f"Running: {len(running)}")
    for job in running:
        print(f"  {job[1] if len(job) > 1 else job[0]}")
    print(f"Pending: {len(pending)}")
    for job in pending:
        print(f"  {job[1] if len(job) > 1 else job[0]}")


def cmd_history(args):
    data = history(args.server, args.prompt_id)
    if args.prompt_id:
        print(json.dumps(data, indent=2)[:2000])
    else:
        for pid, info in list(data.items())[-10:]:
            status = info.get("status", {}).get("status_str", "?")
            print(f"  {pid}: {status}")


def cmd_clear(args):
    clear_queue(args.server)
    print("Queue cleared")


def cmd_cancel(args):
    cancel_job(args.server, args.prompt_id)
    print(f"Cancelled: {args.prompt_id}")


def cmd_interrupt(args):
    interrupt(args.server)
    print("Interrupted current job")


def cmd_watch(args):
    client_id = str(uuid.uuid4())
    watch(args.server, client_id, args.prompt_id)


def main():
    parser = argparse.ArgumentParser(description="ComfyUI API CLI")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="ComfyUI address (default 127.0.0.1:8188)")
    sp = parser.add_subparsers(dest="cmd", required=True)

    sp.add_parser("status", help="Health check + GPU info").set_defaults(func=cmd_status)

    p_nodes = sp.add_parser("nodes", help="List node types")
    p_nodes.add_argument("--grep", help="Filter by substring")
    p_nodes.set_defaults(func=cmd_nodes)

    p_schema = sp.add_parser("schema", help="Show node schema")
    p_schema.add_argument("node", help="Node type name")
    p_schema.set_defaults(func=cmd_schema)

    p_submit = sp.add_parser("submit", help="Submit workflow")
    p_submit.add_argument("workflow", help="Workflow JSON file")
    p_submit.add_argument("--watch", action="store_true", help="Monitor execution")
    p_submit.add_argument("--batch", type=int, default=1, help="Submit N copies with random seeds")
    p_submit.add_argument("--seed-node", default="18", help="Node ID containing noise_seed")
    p_submit.set_defaults(func=cmd_submit)

    sp.add_parser("queue", help="Show queue").set_defaults(func=cmd_queue)

    p_hist = sp.add_parser("history", help="Job history")
    p_hist.add_argument("prompt_id", nargs="?", help="Specific prompt_id (optional)")
    p_hist.set_defaults(func=cmd_history)

    sp.add_parser("clear", help="Clear pending queue").set_defaults(func=cmd_clear)

    p_cancel = sp.add_parser("cancel", help="Cancel a job")
    p_cancel.add_argument("prompt_id")
    p_cancel.set_defaults(func=cmd_cancel)

    sp.add_parser("interrupt", help="Interrupt running job").set_defaults(func=cmd_interrupt)

    p_watch = sp.add_parser("watch", help="Monitor a job")
    p_watch.add_argument("prompt_id")
    p_watch.set_defaults(func=cmd_watch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
