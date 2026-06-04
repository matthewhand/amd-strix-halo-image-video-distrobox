"""SSRF guard for user-supplied URLs the server fetches (LLM base_url, etc.).

Tailored to this app: legit endpoints are loopback (Ollama), LAN (LM Studio),
*and* public (OpenAI/OpenRouter), so we don't blanket-block private hosts.
We only reject what is never a legit LLM endpoint and is dangerous to fetch:
non-http(s) schemes (file://, gopher://…), cloud metadata, and
reserved/multicast/unspecified addresses. DNS is resolved so a hostname can't
rebind to a blocked IP.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_METADATA_V4 = ipaddress.ip_network("169.254.0.0/16")   # link-local incl. 169.254.169.254
_METADATA_V6 = ipaddress.ip_network("fd00:ec2::/64")    # IMDSv6


def _blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_multicast or ip.is_reserved or ip.is_unspecified
        or (ip.version == 4 and ip in _METADATA_V4)
        or (ip.version == 6 and ip in _METADATA_V6)
    )


def validate_llm_base_url(url: str) -> str:
    """Return `url` if it's safe to fetch server-side; raise ValueError otherwise.

    Allows http/https to loopback, private (LAN) and public hosts; blocks other
    schemes and cloud-metadata / reserved / multicast addresses.
    """
    u = urlparse((url or "").strip())
    if u.scheme not in ("http", "https"):
        raise ValueError("base_url scheme must be http or https")
    host = u.hostname
    if not host:
        raise ValueError("base_url is missing a host")
    port = u.port or (443 if u.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise ValueError(f"base_url DNS resolution failed: {exc}")
    if not infos:
        raise ValueError("base_url did not resolve")
    for info in infos:
        sockaddr = info[4]
        ip = ipaddress.ip_address(sockaddr[0])
        if _blocked_ip(ip):
            raise ValueError("base_url resolves to a blocked address (metadata/reserved)")
    return url
