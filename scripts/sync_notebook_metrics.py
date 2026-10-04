import argparse
import ast
import difflib
import sys
from pathlib import Path

from _notebook_utils import cell_source, code_cells, read_notebook, set_cell_source, write_notebook

ROOT = Path(__file__).resolve().parents[1]
SOURCE_OF_TRUTH = ROOT / "src" / "utils" / "metrics.py"
NOTEBOOK_TARGETS = [
    ROOT / "kaggle" / "notebooks" / "03_statistical_models.ipynb",
    ROOT / "kaggle" / "notebooks" / "04_ml_models.ipynb",
]
SYNCED_FUNCTIONS = ("mape", "wape", "mase")


def function_spans(text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    lines = text.split("\n")
    return {
        node.name: (node.lineno - 1, node.end_lineno, "\n".join(lines[node.lineno - 1:node.end_lineno]))
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in SYNCED_FUNCTIONS
    }


def sync_notebook(nb_path, canonical, check_only):
    nb = read_notebook(nb_path)
    changed = []
    diffs = []
    found = set()

    for cell in code_cells(nb):
        original = cell_source(cell)
        spans = function_spans(original)
        if not spans:
            continue
        found |= set(spans)

        lines = original.split("\n")
        for name, (start, end, current) in sorted(spans.items(), key=lambda kv: -kv[1][0]):
            if current != canonical[name]:
                changed.append(name)
                lines[start:end] = canonical[name].split("\n")
        updated = "\n".join(lines)

        if updated != original:
            diffs.append("\n".join(difflib.unified_diff(
                original.splitlines(), updated.splitlines(),
                fromfile=f"{nb_path}::cell", tofile=f"{nb_path}::cell (synced)", lineterm="",
            )))
            if not check_only:
                set_cell_source(cell, updated)

    missing = set(canonical) - found
    if missing:
        raise SystemExit(f"{nb_path}: missing function definition(s) for {sorted(missing)}")

    if changed and check_only:
        print(f"DRIFT in {nb_path} for: {', '.join(sorted(set(changed)))}\n" + "\n".join(diffs) + "\n")
    elif changed:
        write_notebook(nb_path, nb)
        print(f"Synced {nb_path}: {', '.join(sorted(set(changed)))}")

    return bool(changed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    canonical = {
        name: segment for name, (_, _, segment) in function_spans(
            SOURCE_OF_TRUTH.read_text(encoding="utf-8")
        ).items()
    }
    if set(canonical) != set(SYNCED_FUNCTIONS):
        raise SystemExit(f"{SOURCE_OF_TRUTH} must define exactly {SYNCED_FUNCTIONS}")

    any_drift = False
    for target in NOTEBOOK_TARGETS:
        if sync_notebook(target, canonical, check_only=args.check):
            any_drift = True

    if args.check:
        if any_drift:
            print("Notebook metrics are out of sync with src/utils/metrics.py. "
                  "Run 'python scripts/sync_notebook_metrics.py' to fix.")
            sys.exit(1)
        print("Notebook metrics match src/utils/metrics.py.")
    elif not any_drift:
        print("Already in sync, nothing to do.")


if __name__ == "__main__":
    main()
