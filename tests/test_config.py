from pathlib import Path

from trading_pipeline.config_manager import load_json_config, save_json_config


def test_save_and_load_config(tmp_path: Path):
    path = tmp_path / "config.json"
    save_json_config({"use_llm": False, "initial_budget": 123}, path)
    assert load_json_config(path)["initial_budget"] == 123
