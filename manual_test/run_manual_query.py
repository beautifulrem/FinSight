from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
import sys

os.environ.setdefault("QI_USE_LIVE_MARKET", "1")
os.environ.setdefault("QI_USE_LIVE_MACRO", "1")
os.environ.setdefault("QI_USE_LIVE_NEWS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.service import build_default_service, clear_service_caches


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    args = _build_parser().parse_args()
    query = args.query.strip() if args.query else _prompt_query()
    if not query:
        raise SystemExit("query is empty")

    clear_service_caches()
    service = build_default_service()

    nlu_result = service.analyze_query(query, debug=True)
    retrieval_result = service.retrieve_evidence(nlu_result, top_k=args.top_k, debug=True)

    run_dir = _prepare_run_dir(query)
    (run_dir / "query.txt").write_text(query + "\n", encoding="utf-8")
    (run_dir / "nlu_result.json").write_text(json.dumps(nlu_result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "retrieval_result.json").write_text(
        json.dumps(retrieval_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "query": query,
                "output_dir": str(run_dir),
                "nlu_result": str(run_dir / "nlu_result.json"),
                "retrieval_result": str(run_dir / "retrieval_result.json"),
            },
            ensure_ascii=False,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one manual Query Intelligence test and write NLU/Retrieval JSON outputs.")
    parser.add_argument("--query", type=str, default="", help="Question to evaluate. If omitted, the script prompts in the terminal.")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top_k passed to the retrieval module.")
    return parser


def _prompt_query() -> str:
    sys.stdout.write("请输入问题: ")
    sys.stdout.flush()
    stdin_buffer = getattr(sys.stdin, "buffer", None)
    if stdin_buffer is None:
        return sys.stdin.readline().strip()
    return _decode_stdin_bytes(stdin_buffer.readline()).strip()


def _decode_stdin_bytes(raw: bytes) -> str:
    encodings = [
        sys.stdin.encoding,
        sys.getfilesystemencoding(),
        "utf-8",
        "gb18030",
        "gbk",
        "big5",
    ]
    tried: set[str] = set()
    for encoding in encodings:
        if not encoding:
            continue
        normalized = encoding.lower()
        if normalized in tried:
            continue
        tried.add(normalized)
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _prepare_run_dir(query: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = _slugify(query)[:48] or "manual-query"
    run_dir = OUTPUT_DIR / f"{timestamp}-{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _slugify(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", "-", lowered)
    lowered = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff_-]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-_")


if __name__ == "__main__":
    main()
