from trading_pipeline.config_manager import load_json_config, save_json_config


def test_json_config_roundtrip(tmp_path):
    path = tmp_path / "nested" / "config.json"
    config = {"budget": 10000, "risk_multiplier_min": 0.25}
    save_json_config(config, path)
    assert load_json_config(path) == config


def test_load_missing_json_config_returns_empty_dict(tmp_path):
    assert load_json_config(tmp_path / "missing.json") == {}
