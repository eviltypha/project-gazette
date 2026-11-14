"""CLI entry point: ``python -m gazette.feedback export | approve <id>``."""

import sys

from gazette.interfaces.cli import feedback_main

if __name__ == "__main__":
    sys.exit(feedback_main())
