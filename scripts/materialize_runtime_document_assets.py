from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.runtime_document_assets import (  # noqa: E402
    DEFAULT_MAX_DOCUMENTS,
    RuntimeDocumentAssetBuilder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize runtime DocumentRetriever assets from retrieval_corpus.jsonl.")
    parser.add_argument("--corpus-path", type=Path, default=ROOT / "data" / "training_assets" / "retrieval_corpus.jsonl")
    parser.add_argument("--output-path", type=Path, default=ROOT / "data" / "runtime" / "documents.jsonl")
    parser.add_argument("--max-documents", type=int, default=DEFAULT_MAX_DOCUMENTS)
    args = parser.parse_args()

    builder = RuntimeDocumentAssetBuilder(corpus_path=args.corpus_path, max_documents=args.max_documents)
    summary = builder.write(args.output_path)
    print(json.dumps(summary.__dict__, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
