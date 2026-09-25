"""Stream the answer text while the LLM is still writing its JSON draft.

The agent asks the model for a JSON object (``{"answer": ..., "key_points": ..., ...}``). To show text
as it is generated, ``AnswerTextStream`` accumulates the raw completion chunks and emits the decoded
characters of the ``"answer"`` string value that have become available, handling escapes that are split
across chunks. Streamed text is a preview only: the final ``answer`` event carries the verified,
compliance-checked answer and replaces it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

_ANSWER_KEY = re.compile(r'"answer"\s*:\s*"')


class AnswerTextStream:
    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._raw = ""
        self._sent = 0
        self._done = False

    def feed(self, chunk: str) -> None:
        if self._done or not chunk:
            return
        self._raw += chunk
        text, complete = _partial_answer(self._raw)
        if text is None:
            return
        if len(text) > self._sent:
            self._emit(text[self._sent :])
            self._sent = len(text)
        self._done = complete

    @property
    def text_sent(self) -> int:
        return self._sent


def _partial_answer(raw: str) -> tuple[str | None, bool]:
    """Decoded prefix of the ``answer`` string value and whether the string is closed."""
    match = _ANSWER_KEY.search(raw)
    if match is None:
        return None, False
    body = raw[match.end() :]
    index = 0
    while index < len(body):
        char = body[index]
        if char == "\\":
            if index + 1 >= len(body):
                break  # escape split across chunks: wait for the next one
            if body[index + 1] == "u" and index + 6 > len(body):
                break
            index += 6 if body[index + 1] == "u" else 2
            continue
        if char == '"':
            return _decode(body[:index]), True
        index += 1
    return _decode(body[:index]), False


def _decode(segment: str) -> str:
    try:
        return json.loads(f'"{segment}"')
    except json.JSONDecodeError:
        return segment


def stream_writer() -> Callable[[Any], None]:
    """LangGraph's custom stream writer, or a no-op outside a streamed run (``invoke``, unit tests)."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # not inside a runnable context
        return lambda _payload: None
