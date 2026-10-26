"""
tests/test_no_gates.py: Gate G0 candidate gates test.
Rule 3 from Section 1.3 & Section 11.9:
Static test that greps backend/app for 'return None' / 'continue' inside
recommend and the rerank loop on any of language, genre, year, decade,
artist, popularity, _trusted. Fails on any hit.
"""
from __future__ import annotations

import ast
import os
import re

GATE_ATTRIBUTES = {
    "language", "genre", "year", "decade", "artist", "popularity", "_trusted",
    "languageaffinity", "min_language_affinity", "_trusted_scene"
}


def _check_ast_for_gates(file_path: str) -> list[str]:
    violations = []
    with open(file_path, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src, filename=file_path)

    for node in ast.walk(tree):
        # Target recommend functions and ranking / reranking loops
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name.lower()
            if any(term in func_name for term in ["recommend", "rerank", "evaluate_candidate", "_find_and_rank"]):
                # Walk statements inside function
                for child in ast.walk(node):
                    if isinstance(child, ast.If):
                        # Check if body contains Continue or Return(None)
                        has_drop = any(
                            isinstance(stmt, ast.Continue) or
                            (isinstance(stmt, ast.Return) and (stmt.value is None or getattr(stmt.value, "value", 1) is None))
                            for stmt in child.body
                        )
                        if has_drop:
                            # Check if the if-condition references any gate attribute
                            cond_str = ast.unparse(child.test).lower()
                            matched_gates = [g for g in GATE_ATTRIBUTES if g in cond_str]
                            if matched_gates:
                                violations.append(
                                    f"[{file_path}:{child.lineno}] Gate found in {node.name}(): "
                                    f"drops candidate on {matched_gates}: '{ast.unparse(child.test)}'"
                                )

    return violations


def test_no_gates_in_backend():
    backend_app_dir = os.path.abspath("backend/app")
    all_violations: list[str] = []

    for root, dirs, files in os.walk(backend_app_dir):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                v = _check_ast_for_gates(p)
                all_violations.extend(v)

    assert not all_violations, (
        f"Candidate gates detected in recommendation path ({len(all_violations)} violations):\n"
        + "\n".join(all_violations)
    )
