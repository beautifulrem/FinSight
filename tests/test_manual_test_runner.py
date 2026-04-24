from __future__ import annotations

import json
from pathlib import Path

from manual_test.run_manual_query import _prepare_run_dir, _slugify


def test_slugify_keeps_chinese_and_ascii() -> None:
    assert _slugify("中国平安 最近 为什么 跌?") == "中国平安-最近-为什么-跌"
    assert _slugify("ETF vs LOF ???") == "etf-vs-lof"


def test_prepare_run_dir_creates_output_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("manual_test.run_manual_query.OUTPUT_DIR", tmp_path / "output")

    run_dir = _prepare_run_dir("中国平安最近为什么跌？")

    assert run_dir.exists()
    assert run_dir.parent == tmp_path / "output"
    assert "中国平安最近为什么跌" in run_dir.name
