from __future__ import annotations

import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.bootstrap_public_data import PublicDataBootstrapper  # noqa: E402
from query_intelligence.integrations.tushare_provider import TushareMarketProvider, TushareNewsProvider  # noqa: E402


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
    token = os.getenv("TUSHARE_TOKEN")
    bootstrapper = PublicDataBootstrapper(
        output_dir=output_dir,
        tushare_market_provider=TushareMarketProvider.from_token(token) if token else None,
        tushare_news_provider=TushareNewsProvider.from_token(token) if token else None,
    )
    bootstrapper.run()
    print(f"public data package written to {output_dir}")
    warnings_path = output_dir / "bootstrap_warnings.json"
    if warnings_path.exists():
        print(f"warnings written to {warnings_path}")


if __name__ == "__main__":
    main()
