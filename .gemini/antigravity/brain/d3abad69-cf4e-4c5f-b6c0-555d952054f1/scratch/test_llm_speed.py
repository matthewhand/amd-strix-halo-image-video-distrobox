import sys

ROOT = "/home/matthewh/amd-strix-halo-image-video-toolboxes"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from slopfinity.llm.probe import ping

def test_speed():
    targets = [
        ("10.0.0.31", 1234, "lmstudio"),
        ("10.0.0.31", 14434, "ollama"),
        ("10.0.0.36", 1234, "lmstudio"),
        ("10.0.0.36", 14434, "ollama"),
    ]
    model_id = "llama3"
    
    for ip, port, provider in targets:
        base_url = f"http://{ip}:{port}/v1"
        print(f"\nPinging {provider} at {base_url}...")
        res = ping(base_url, provider, model_id, timeout=10)
        print(f"Result: {res}")

if __name__ == "__main__":
    test_speed()
