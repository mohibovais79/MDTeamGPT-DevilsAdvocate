# MODIFIED: added environment-variable override for credentials, and two
# new default config keys for the experimental toggles. See EXPERIMENTS.md.
#
# SECURITY NOTE: never put a real API key in config.json if this repo (or
# your fork of it) is public. Set the MDT_API_KEY environment variable
# instead -- it always takes precedence over whatever is in config.json.

import json
import os

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "api_key": "",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "text_model": "qwen-plus",
    "vl_model": "qwen-vl-plus",
    "enable_tools": True,
    "enable_devils_advocate": False,
    "enable_conflict_tools": False,
}


def load_config():
    config = dict(DEFAULT_CONFIG)

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception:
            pass

    # Environment variables always win over the config file, so API keys
    # never need to be written to disk / committed to version control.
    env_key = os.environ.get("MDT_API_KEY")
    if env_key:
        config["api_key"] = env_key

    env_base_url = os.environ.get("MDT_BASE_URL")
    if env_base_url:
        config["base_url"] = env_base_url

    return config


def save_config(config_dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        return True
    except Exception as e:
        print(f"Save config failed: {e}")
        return False