from __future__ import annotations

from manual_test.run_manual_query import _decode_stdin_bytes


def test_decode_stdin_bytes_accepts_gb18030_chinese_input() -> None:
    raw = "你好，你觉得 寒武纪怎么样\n".encode("gb18030")

    assert _decode_stdin_bytes(raw).strip() == "你好，你觉得 寒武纪怎么样"
