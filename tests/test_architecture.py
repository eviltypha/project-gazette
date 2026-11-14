"""Enforces the Clean Architecture dependency rule by inspecting imports.

domain  <-  application  <-  infrastructure / interfaces  <-  container / entry points
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "gazette"

FORBIDDEN = {
    "domain": ("gazette.application", "gazette.infrastructure", "gazette.interfaces",
               "gazette.container", "fastapi", "sqlite3", "numpy", "rank_bm25"),
    "application": ("gazette.infrastructure", "gazette.interfaces", "gazette.container",
                    "fastapi", "sqlite3", "rank_bm25"),
    "infrastructure": ("gazette.interfaces", "gazette.container", "fastapi"),
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layer_does_not_depend_on_outer_layers(layer):
    violations = [
        f"{py.relative_to(PACKAGE)} imports {name}"
        for py in (PACKAGE / layer).rglob("*.py")
        for name in _imports(py)
        if name.startswith(FORBIDDEN[layer])
    ]
    assert not violations, violations
