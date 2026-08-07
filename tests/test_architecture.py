from __future__ import annotations

import ast
from pathlib import Path


def test_package_does_not_import_foundry_or_publication_layers() -> None:
    root = Path(__file__).parents[1] / "src" / "congreso_open_data"
    forbidden = {"cpl_data_foundry", "materialize", "gold", "serving", "postgres"}
    violations: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = set(node.module.split("."))
                if parts & forbidden:
                    violations.append(f"{path.relative_to(root)}:{node.lineno}:{node.module}")
    assert violations == []
