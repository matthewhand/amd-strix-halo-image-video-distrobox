"""SSRF guard for user-supplied LLM base_url + AudioWorker import resolution."""
import pytest
from slopfinity.net_guard import validate_llm_base_url


@pytest.mark.parametrize("bad", [
    "http://169.254.169.254/latest/meta-data/",   # cloud metadata
    "file:///etc/passwd",                          # non-http scheme
    "gopher://internal/",                          # non-http scheme
    "http://224.0.0.1/",                           # multicast
    "ftp://host/x",                                # non-http scheme
    "",                                            # empty
    "http://",                                     # no host
])
def test_blocks_dangerous(bad):
    with pytest.raises(ValueError):
        validate_llm_base_url(bad)


@pytest.mark.parametrize("ok", [
    "http://127.0.0.1:11434/v1",   # loopback (ollama)
    "http://10.0.0.107:1234/v1",   # LAN (LM Studio)
    "https://api.openai.com/v1",   # public (openai-compat)
])
def test_allows_legit(ok):
    assert validate_llm_base_url(ok) == ok


def test_audio_worker_import_resolves():
    # AudioWorker.run_stage imports this; it must come from worker_sh, not workers.
    from slopfinity.worker_sh import run_audio_heartmula
    assert callable(run_audio_heartmula)
    with pytest.raises(ImportError):
        from slopfinity.workers import run_audio_heartmula  # noqa: F401
