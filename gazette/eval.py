"""CLI entry point: ``python -m gazette.eval [--min-recall 0.80]``."""

import sys

from gazette.interfaces.cli import eval_main

if __name__ == "__main__":
    sys.exit(eval_main())
