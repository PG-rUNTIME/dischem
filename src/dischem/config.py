"""Configuration loading. One YAML file, overridable from the CLI with key=value pairs."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "data_dir": "data",
    "images_dir": "data/images",          # folder that contains img_XXXXXX.jpg
    "output_dir": "outputs",
    "cache_dir": "cache",
    "input_file": "Test_Masked.csv",
    "llm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "EMPTY",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "concurrency": 16,
        "request_timeout_s": 180,
        "max_retries": 1,                 # challenge rule: at most one retry per failed call
        "temperature": 0.0,
        "extra_body": {},                 # e.g. {"chat_template_kwargs": {"enable_thinking": false}}
    },
    "images": {
        "enabled": True,
        "max_per_record": 3,
        "max_side_px": 1024,
        "jpeg_quality": 90,
        "max_tokens": 700,
    },
    "compose": {"max_tokens": 1400, "few_shot": 2, "max_source_chars": 3500},
    "verify": {"enabled": True, "max_tokens": 500},
    "siblings": {                         # optional catalogue lookup, off by default (see README)
        "enabled": False,
        "min_siblings": 2,
        "min_name_jaccard": 0.5,
        "fields": ["Packaging Type", "Application Area", "Recyclable Packaging", "Certifications"],
    },
    "budget": {"max_runtime_minutes": 170, "reserve_minutes": 10},
}


def _deep_update(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as fh:
            _deep_update(cfg, yaml.safe_load(fh) or {})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key=value, got {item!r}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    for k in ("data_dir", "images_dir", "output_dir", "cache_dir"):
        cfg[k] = str(Path(cfg[k]))
    return cfg
