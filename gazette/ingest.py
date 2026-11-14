"""CLI entry point: ``python -m gazette.ingest`` builds BM25 + vector indexes into index/."""

import sys

from gazette.config import Settings
from gazette.container import run_ingest


def main() -> int:
    settings = Settings.from_env()
    report = run_ingest(settings)
    print(
        f"Indexed {report.paragraphs} paragraphs from {report.circulars} circulars "
        f"({report.amending_circulars} amending) into {settings.index_dir}/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
