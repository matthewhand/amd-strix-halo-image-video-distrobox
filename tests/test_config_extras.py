"""DEFAULT_CONFIG role keys + save_queue stable-id stamping."""


def test_default_config_role_keys_and_qwen_base():
    from slopfinity.config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["base_model"] == "qwen"
    assert DEFAULT_CONFIG["audio_model"] == "none"
    assert DEFAULT_CONFIG["tts_model"] == "none"
    assert DEFAULT_CONFIG["upscale_model"] == "none"


def test_save_queue_stamps_stable_id():
    # run_fleet appends rows without 'id'; save_queue must stamp a stable id so
    # the id doesn't churn (fresh uuid) on every save.
    from slopfinity import config as cfg
    try:
        item = {"prompt": "idtest", "ts": 123.0, "status": "pending"}
        cfg.save_queue([item])
        assert item.get("id")
        first = item["id"]
        cfg.save_queue([item])
        assert item["id"] == first
    finally:
        cfg.save_queue([])
