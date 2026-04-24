from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.runtime_entity_assets import (  # noqa: E402
    RuntimeEntityAssetBuilder,
    fetch_akshare_security_universe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize runtime entity/alias assets from public training assets.")
    parser.add_argument("--seed-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--training-assets-dir", type=Path, default=ROOT / "data" / "training_assets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "runtime")
    parser.add_argument("--max-training-pairs", type=int, default=80_000)
    parser.add_argument("--no-akshare", action="store_true", help="Skip live AKShare universe fetch and use local training assets only.")
    args = parser.parse_args()

    seed_entities = _read_csv(args.seed_dir / "entity_master.csv")
    seed_aliases = _read_csv(args.seed_dir / "alias_table.csv")
    extra_rows = []
    if not args.no_akshare:
        print(json.dumps({"stage": "fetch_akshare_universe", "status": "starting"}, ensure_ascii=False), flush=True)
        extra_rows = fetch_akshare_security_universe()
        print(json.dumps({"stage": "fetch_akshare_universe", "rows": len(extra_rows)}, ensure_ascii=False), flush=True)

    print(json.dumps({"stage": "build_runtime_entities", "status": "starting"}, ensure_ascii=False), flush=True)
    builder = RuntimeEntityAssetBuilder(
        seed_entities=seed_entities,
        seed_aliases=seed_aliases,
        training_assets_dir=args.training_assets_dir,
        max_training_pairs=args.max_training_pairs,
    )
    summary = builder.write(args.output_dir, extra_universe_rows=extra_rows)
    print(json.dumps(summary.__dict__, ensure_ascii=False), flush=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
