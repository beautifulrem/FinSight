"""Chatbot configuration: defaults, config file, .env and environment overrides."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CONFIG_PATH = ROOT / "config" / "app_config.json"


DEFAULT_CHATBOT_CONFIG: dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
    },
    "ui": {
        "title": "FinSight Financial Research Assistant",
        "input_placeholder": "Ask a financial question, e.g. What do you think about Ping An Insurance (601318.SH)?",
        "submit_text": "Submit",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "chat_path": "/chat/completions",
        "model": "deepseek-v4-flash",
        "api_key": "",
        "timeout_seconds": 60,
        "thinking_type": "enabled",
        "reasoning_effort": "high",
        "max_tokens": 8192,
    },
    "live_data": {
        "enabled": True,
    },
}


def load_chatbot_config(
    config_path: str | Path | None = None,
    *,
    load_env_file: bool = True,
) -> dict[str, Any]:
    if load_env_file:
        _load_dotenv(ROOT / ".env")

    path = Path(os.getenv("FINANCIAL_CHATBOT_CONFIG") or config_path or DEFAULT_CONFIG_PATH)
    config = copy.deepcopy(DEFAULT_CHATBOT_CONFIG)
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"chatbot config must be a JSON object: {path}")
        _deep_merge(config, loaded)

    _apply_env_overrides(config)
    _coerce_config_types(config)
    return config


def apply_live_data_env(config: dict[str, Any]) -> None:
    enabled = bool((config.get("live_data") or {}).get("enabled", True))
    value = "1" if enabled else "0"
    for name in (
        "QI_USE_LIVE_MARKET",
        "QI_USE_LIVE_NEWS",
        "QI_USE_LIVE_ANNOUNCEMENT",
        "QI_USE_LIVE_MACRO",
    ):
        os.environ.setdefault(name, value)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _apply_env_overrides(config: dict[str, Any]) -> None:
    mappings = {
        "CHATBOT_HOST": ("server", "host"),
        "CHATBOT_PORT": ("server", "port"),
        "CHATBOT_TITLE": ("ui", "title"),
        "CHATBOT_INPUT_PLACEHOLDER": ("ui", "input_placeholder"),
        "CHATBOT_SUBMIT_TEXT": ("ui", "submit_text"),
        "DEEPSEEK_BASE_URL": ("deepseek", "base_url"),
        "DEEPSEEK_CHAT_PATH": ("deepseek", "chat_path"),
        "DEEPSEEK_MODEL": ("deepseek", "model"),
        "DEEPSEEK_API_KEY": ("deepseek", "api_key"),
        "DEEPSEEK_TIMEOUT_SECONDS": ("deepseek", "timeout_seconds"),
        "DEEPSEEK_THINKING_TYPE": ("deepseek", "thinking_type"),
        "DEEPSEEK_REASONING_EFFORT": ("deepseek", "reasoning_effort"),
        "DEEPSEEK_MAX_TOKENS": ("deepseek", "max_tokens"),
        "CHATBOT_LIVE_DATA": ("live_data", "enabled"),
    }
    for env_name, path in mappings.items():
        if env_name in os.environ:
            section, key = path
            config.setdefault(section, {})[key] = os.environ[env_name]


def _coerce_config_types(config: dict[str, Any]) -> None:
    config["server"]["port"] = int(config["server"]["port"])
    config["deepseek"]["timeout_seconds"] = int(config["deepseek"]["timeout_seconds"])
    max_tokens = config["deepseek"].get("max_tokens")
    config["deepseek"]["max_tokens"] = int(max_tokens) if max_tokens not in {None, ""} else None
    config["live_data"]["enabled"] = _parse_bool(config["live_data"].get("enabled", True))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
