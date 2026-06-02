"""Hermetic tests for the SSRF guard on the user-supplied llm.base_url.

`_validate_llm_base_url` is the security boundary that stops a hostile
(or XSS-injected same-origin) /settings POST from repointing the
dashboard's LLM calls at cloud-metadata services, RFC1918 internal
hosts, or non-http(s) schemes (file://, gopher://, ...).

These tests are fully hermetic: DNS resolution is monkeypatched so no
real network calls happen, and `cfg.load_config` is stubbed so the
`allow_cloud_endpoints` toggle is controlled per-test.
"""
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import slopfinity.server as server  # noqa: E402


@pytest.fixture
def cloud_allowed(monkeypatch):
    """allow_cloud_endpoints=True so non-loopback hosts reach IP screening."""
    monkeypatch.setattr(
        server.cfg, "load_config", lambda: {"allow_cloud_endpoints": True}
    )


@pytest.fixture
def cloud_blocked(monkeypatch):
    """allow_cloud_endpoints=False (the safe default)."""
    monkeypatch.setattr(
        server.cfg, "load_config", lambda: {"allow_cloud_endpoints": False}
    )


def _fixed_resolver(mapping):
    """Build a hermetic _resolve_host replacement.

    `mapping` maps hostname -> list of IPs. A literal IP host resolves to
    itself, mirroring getaddrinfo's behaviour for IP literals.
    """

    def _resolve(host):
        if host in mapping:
            return mapping[host]
        # IP literals resolve to themselves.
        return [host]

    return _resolve


# --------------------------------------------------------------------------
# Scheme validation (independent of the cloud toggle)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://127.0.0.1:6379/_INFO",
        "ftp://internal/secrets",
        "data:text/plain,hi",
    ],
)
def test_non_http_schemes_blocked(url, cloud_allowed):
    ok, err = server._validate_llm_base_url(url)
    assert ok is False
    assert "scheme" in err


def test_empty_url_blocked():
    ok, err = server._validate_llm_base_url("")
    assert ok is False


# --------------------------------------------------------------------------
# Cloud-metadata endpoints — always blocked, even with cloud enabled
# --------------------------------------------------------------------------


def test_metadata_ip_blocked(cloud_allowed, monkeypatch):
    monkeypatch.setattr(server, "_resolve_host", _fixed_resolver({}))
    ok, err = server._validate_llm_base_url("http://169.254.169.254/latest/meta-data/")
    assert ok is False
    assert "metadata" in err or "link-local" in err


def test_metadata_hostname_blocked(cloud_allowed):
    ok, err = server._validate_llm_base_url(
        "http://metadata.google.internal/computeMetadata/v1/"
    )
    assert ok is False
    assert "metadata" in err


def test_alibaba_metadata_ip_blocked(cloud_allowed, monkeypatch):
    monkeypatch.setattr(server, "_resolve_host", _fixed_resolver({}))
    ok, err = server._validate_llm_base_url("http://100.100.100.200/")
    assert ok is False


def test_link_local_range_blocked(cloud_allowed, monkeypatch):
    """Any 169.254.0.0/16 address, not just the canonical metadata IP."""
    monkeypatch.setattr(server, "_resolve_host", _fixed_resolver({}))
    ok, err = server._validate_llm_base_url("http://169.254.1.1/")
    assert ok is False
    assert "link-local" in err or "metadata" in err


# --------------------------------------------------------------------------
# RFC1918 private ranges — blocked when reached via cloud endpoints
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "10.0.0.5",  # 10/8
        "10.255.255.254",
        "172.16.0.1",  # 172.16/12
        "172.31.255.254",
        "192.168.1.1",  # 192.168/16
        "192.168.255.254",
    ],
)
def test_rfc1918_ranges_blocked(ip, cloud_allowed, monkeypatch):
    monkeypatch.setattr(server, "_resolve_host", _fixed_resolver({}))
    ok, err = server._validate_llm_base_url(f"http://{ip}/v1")
    assert ok is False, f"{ip} should be blocked"
    assert "private" in err or "RFC1918" in err


# --------------------------------------------------------------------------
# DNS rebinding — a public hostname that resolves to an internal IP
# --------------------------------------------------------------------------


def test_dns_rebinding_to_metadata_blocked(cloud_allowed, monkeypatch):
    monkeypatch.setattr(
        server,
        "_resolve_host",
        _fixed_resolver({"evil.example.com": ["169.254.169.254"]}),
    )
    ok, err = server._validate_llm_base_url("https://evil.example.com/v1")
    assert ok is False


def test_dns_rebinding_to_rfc1918_blocked(cloud_allowed, monkeypatch):
    monkeypatch.setattr(
        server,
        "_resolve_host",
        _fixed_resolver({"sneaky.example.com": ["10.0.0.5"]}),
    )
    ok, err = server._validate_llm_base_url("https://sneaky.example.com/v1")
    assert ok is False
    assert "private" in err or "RFC1918" in err


def test_dns_rebinding_mixed_addresses_blocked(cloud_allowed, monkeypatch):
    """If ANY resolved address is internal, reject the whole host."""
    monkeypatch.setattr(
        server,
        "_resolve_host",
        _fixed_resolver({"multi.example.com": ["8.8.8.8", "192.168.0.1"]}),
    )
    ok, err = server._validate_llm_base_url("https://multi.example.com/v1")
    assert ok is False


# --------------------------------------------------------------------------
# Valid public endpoints — allowed when cloud endpoints are enabled
# --------------------------------------------------------------------------


def test_valid_public_https_allowed(cloud_allowed, monkeypatch):
    monkeypatch.setattr(
        server,
        "_resolve_host",
        _fixed_resolver({"api.openai.com": ["104.18.7.192"]}),
    )
    ok, err = server._validate_llm_base_url("https://api.openai.com/v1")
    assert ok is True, err
    assert err == ""


# --------------------------------------------------------------------------
# Localhost — always allowed for local LLM providers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:1234/v1",
        "http://127.0.0.1:11434/v1",
        "http://[::1]:8080/v1",
    ],
)
def test_localhost_allowed_for_local(url, cloud_blocked):
    ok, err = server._validate_llm_base_url(url)
    assert ok is True, err


def test_loopback_allowed_without_dns(cloud_blocked, monkeypatch):
    """Loopback must short-circuit before any DNS resolution happens."""

    def _boom(host):  # pragma: no cover - should never be called
        raise AssertionError("DNS resolution attempted for loopback host")

    monkeypatch.setattr(server, "_resolve_host", _boom)
    ok, err = server._validate_llm_base_url("http://localhost:1234/v1")
    assert ok is True


# --------------------------------------------------------------------------
# Default (cloud disabled): non-loopback hosts rejected outright
# --------------------------------------------------------------------------


def test_non_loopback_blocked_when_cloud_disabled(cloud_blocked):
    ok, err = server._validate_llm_base_url("https://api.openai.com/v1")
    assert ok is False
    assert "Allow Cloud Endpoints" in err
