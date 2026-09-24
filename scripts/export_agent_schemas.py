"""Write JSON Schemas for the agent API from the Pydantic models in ``query_intelligence.contracts``.

python -m scripts.export_agent_schemas          # rewrite schemas/agent_*.schema.json
python -m scripts.export_agent_schemas --check  # exit 1 if the committed files are stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from query_intelligence.contracts import AgentChatRequest, AgentChatResponse, AgentResumeRequest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"
MODELS = {
    "agent_chat_request.schema.json": AgentChatRequest,
    "agent_resume_request.schema.json": AgentResumeRequest,
    "agent_chat_response.schema.json": AgentChatResponse,
}


def render(model) -> str:
    return json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only verify the committed schemas are current")
    args = parser.parse_args(argv)
    stale = []
    for name, model in MODELS.items():
        path = SCHEMA_DIR / name
        text = render(model)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            path.write_text(text, encoding="utf-8")
    if stale:
        print("stale schemas (run python -m scripts.export_agent_schemas): " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
