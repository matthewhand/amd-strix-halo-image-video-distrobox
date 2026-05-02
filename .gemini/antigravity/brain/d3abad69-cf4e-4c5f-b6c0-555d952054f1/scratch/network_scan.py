import asyncio
import socket
import json
import urllib.request
import urllib.error

async def check_port(ip, port, timeout=0.5):
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port),
            timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except:
        return False

async def identify_service(ip, port):
    if port == 1234:
        url = f"http://{ip}:{port}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                data = json.loads(r.read())
                models = [m.get("id") for m in data.get("data", [])]
                return f"LMStudio: {models}"
        except:
            return "LMStudio (unresponsive/no-openai-api)"
    elif port == 11434:
        url = f"http://{ip}:{port}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                data = json.loads(r.read())
                models = [m.get("name") for m in data.get("models", [])]
                return f"Ollama: {models}"
        except:
            return "Ollama (unresponsive)"
    return "Unknown"

async def scan_ip(ip):
    results = []
    for port in [1234, 11434]:
        if await check_port(ip, port):
            info = await identify_service(ip, port)
            results.append((port, info))
    return results

async def main():
    # We prioritize .33 and .133 but scan the whole /24
    ips = [f"10.0.0.{i}" for i in range(1, 255)]
    
    # Sort to put .33 and .133 first if they exist
    # But user specifically said "start with .33 and 10.0.0.133"
    priority = ["10.0.0.33", "10.0.0.133"]
    ips = priority + [ip for ip in ips if ip not in priority]

    print(f"Scanning 10.0.0.0/24 for ports 1234 and 11434...")
    
    # Process in batches of 50 to avoid socket exhaustion
    batch_size = 50
    for i in range(0, len(ips), batch_size):
        batch = ips[i:i+batch_size]
        tasks = [scan_ip(ip) for ip in batch]
        batch_results = await asyncio.gather(*tasks)
        
        for ip, res in zip(batch, batch_results):
            if res:
                for port, info in res:
                    print(f"[FOUND] {ip}:{port} -> {info}")

if __name__ == "__main__":
    asyncio.run(main())
