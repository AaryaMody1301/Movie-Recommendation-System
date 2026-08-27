"""Fail CI if pickle deserialization appears outside approved local model loaders."""

from __future__ import annotations

import ast
from pathlib import Path
import sys


ALLOWED = {
    ("models/content_based.py", "ContentBasedRecommender._load_cache"),
    ("models/content_based.py", "ContentBasedRecommender.load"),
    ("models/collaborative_filtering.py", "CollaborativeRecommender.load"),
    ("models/hybrid_recommender.py", "HybridRecommender.load"),
}
SKIP_PARTS = {".git", ".venv", "venv", "instance", "tests", "__pycache__"}


class PickleLoadVisitor(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path = path
        self.scope = []
        self.calls = []

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pickle"
        ):
            self.calls.append((self.path, ".".join(self.scope), node.lineno))
        self.generic_visit(node)


def main() -> int:
    violations = []
    observed = set()
    for path in sorted(Path(".").rglob("*.py")):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        relative = path.as_posix().removeprefix("./")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        visitor = PickleLoadVisitor(relative)
        visitor.visit(tree)
        for file_name, scope, lineno in visitor.calls:
            key = (file_name, scope)
            observed.add(key)
            if key not in ALLOWED:
                violations.append((file_name, scope, lineno))

    missing = sorted(ALLOWED - observed)
    if violations or missing:
        for file_name, scope, lineno in violations:
            print(f"Unexpected pickle.load at {file_name}:{lineno} ({scope})", file=sys.stderr)
        for file_name, scope in missing:
            print(f"Expected approved pickle loader not found: {file_name} ({scope})", file=sys.stderr)
        return 1

    print("Pickle deserialization is restricted to approved local model/cache loaders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
