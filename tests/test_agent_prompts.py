"""Prompt registry: versions are pinned by hash, system prompts stay cache-friendly."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from query_intelligence.agent import prompts
from query_intelligence.agent.prompts import PROMPTS, get_prompt, prompt_refs

LOCK = Path(prompts.__file__).with_name("prompts.lock.json")


def test_every_prompt_version_matches_the_lock_file():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    current = {
        prompt_id: {version: p.sha for version, p in versions.items()} for prompt_id, versions in PROMPTS.items()
    }

    # Editing a prompt's text without adding a new version (and re-running the eval) fails here.
    assert current == lock, "prompt text changed: add a new version and update prompts.lock.json"


def test_versions_are_immutable_and_distinct():
    for versions in PROMPTS.values():
        shas = [p.sha for p in versions.values()]
        assert len(shas) == len(set(shas)), "two versions of the same prompt have identical text"


def test_system_prompts_have_no_unrendered_placeholders():
    # System prompts are constants (the cacheable prefix); per-request data goes to the user message.
    for versions in PROMPTS.values():
        for p in versions.values():
            head = p.text.split("<output_format>")[0].split("{answer")[0]
            assert not re.search(r"\{[a-z_]+\}", head), f"unrendered template in {p.ref}"


def test_active_version_selection(monkeypatch):
    monkeypatch.delenv("QI_PROMPT_VERSION", raising=False)
    assert get_prompt("agent_system").version == prompts.DEFAULT_PROMPT_VERSION
    monkeypatch.setenv("QI_PROMPT_VERSION", "v2")
    assert get_prompt("agent_system").text.startswith("<role>")
    assert prompt_refs()["compose_system"].startswith("compose_system@v2#")
    monkeypatch.setenv("QI_PROMPT_VERSION", "v9")
    with pytest.raises(KeyError):
        get_prompt("agent_system")


def test_v2_prompts_keep_the_non_negotiable_rules():
    for prompt_id in PROMPTS:
        text = get_prompt(prompt_id, "v2").text
        assert "buy/sell/hold" in text and "untrusted" in text
        assert "evidence id" in text and "Reason:" in text
        example = json.loads(text.split("for example:\n", 1)[1].split("\n</output_format>")[0])
        assert set(example) == {"answer", "key_points", "evidence_used", "limitations"}


def test_user_messages_serialize_with_sorted_keys():
    content = prompts.compose_user_message("q", {"question_style": "fact"}, [{"b": 1, "a": 2}], [], language="zh")
    payload = json.loads(content)
    assert list(payload) == sorted(payload)
    assert '"a": 2, "b": 1' in content


def test_v4_adds_the_document_content_rules_and_is_the_default(monkeypatch):
    monkeypatch.delenv("QI_PROMPT_VERSION", raising=False)
    # v4 became the default after the v3/v4 A/B on test v3 (evaluation/results/ablation-ab-prompt-v*-testv3.json)
    assert prompts.DEFAULT_PROMPT_VERSION == "v4"
    for prompt_id in PROMPTS:
        v3, v4 = get_prompt(prompt_id, "v3").text, get_prompt(prompt_id, "v4").text
        added = v4.replace(prompts._V4_DOCUMENT_RULES + "\n", "")
        assert added == v3, "v4 is v3 plus the document-content rules only"
        assert "contact details" in v4 and "未经其他来源证实" in v4 and "delisting" in v4
    monkeypatch.setenv("QI_PROMPT_VERSION", "v4")
    assert prompt_refs()["compose_system"].startswith("compose_system@v4#")
