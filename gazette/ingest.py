"""CLI entry point: ``python -m gazette.ingest`` builds BM25 + vector indexes into index/."""

import sys

from gazette.interfaces.cli import ingest_main

if __name__ == "__main__":
    sys.exit(ingest_main())
