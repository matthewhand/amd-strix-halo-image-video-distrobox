import json
import urllib.request


def lmstudio_call(sys_p, user_p):
    try:
        req = urllib.request.Request("http://localhost:1234/v1/models")
        with urllib.request.urlopen(req, timeout=5) as r:
            m_id = [m["id"] for m in json.loads(r.read())["data"] if "embed" not in m["id"].lower()][0]
        payload = {
            "model": m_id,
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p},
            ],
            "temperature": 0.7,
        }
        req = urllib.request.Request(
            "http://localhost:1234/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"
