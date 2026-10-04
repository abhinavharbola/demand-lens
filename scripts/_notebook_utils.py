import ast
import json
from pathlib import Path


def read_notebook(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_notebook(path, nb):
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def cell_source(cell):
    return "".join(cell["source"])


def code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def concatenated_source(nb):
    return "\n\n".join(cell_source(c) for c in code_cells(nb))


def strip_shell_lines(source):
    return "\n".join(l for l in source.split("\n") if not l.lstrip().startswith(("!", "%")))


def parse_notebook(nb):
    source = strip_shell_lines(concatenated_source(nb))
    return source, ast.parse(source)


def set_cell_source(cell, text):
    lines = text.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def extract_functions(notebook_path, names):
    source, tree = parse_notebook(read_notebook(notebook_path))
    wanted = set(names)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            found[node.name] = ast.get_source_segment(source, node)
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"{notebook_path}: could not find function(s) {sorted(missing)}")
    return found


def top_level_constants(tree):
    constants = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
    return constants


def function_defaults(tree, func_name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            args = node.args
            positional = args.posonlyargs + args.args
            names = [a.arg for a in positional[len(positional) - len(args.defaults):]]
            values = []
            for default in args.defaults:
                try:
                    values.append(ast.literal_eval(default))
                except ValueError:
                    values.append(None)
            return dict(zip(names, values))
    raise KeyError(func_name)
