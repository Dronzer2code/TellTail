"""Check a source file against Python 3.11 grammar + f-string restrictions.

SiS pins Python 3.11. PEP 701 (3.12) legalised newlines, reused quotes and
backslashes inside f-string expressions; ast.feature_version does not reject
them, so they are checked by hand here.
"""
import ast
import io
import builtins
import sys
import tokenize
from pathlib import Path

BACKSLASH = chr(92)


def check(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    try:
        ast.parse(src, feature_version=(3, 11))
        print(f"OK   {path.name}: parses under Python 3.11 grammar")
    except SyntaxError as e:
        print(f"FAIL {path.name}: 3.11 parse, line {e.lineno}: {e.msg}")
        return 1

    bad = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.STRING:
            continue
        s = tok.string
        n = 0
        while n < len(s) and s[n] not in "\"'":
            n += 1
        prefix = s[:n].lower()
        if "f" not in prefix:
            continue
        rest = s[n:]
        if rest.startswith('"""') or rest.startswith("'''"):
            q = rest[:3]
        else:
            q = rest[0]
        body = rest[len(q):-len(q)]

        depth, expr, i = 0, "", 0
        while i < len(body):
            c = body[i]
            if c == "{":
                if i + 1 < len(body) and body[i + 1] == "{":
                    i += 2
                    continue
                depth += 1
                expr = ""
                i += 1
                continue
            if c == "}" and depth:
                depth -= 1
                if "\n" in expr:
                    bad.append((tok.start[0], "newline inside f-string expression", expr))
                if len(q) == 1 and q in expr:
                    bad.append((tok.start[0], f"delimiter {q} reused inside expression", expr))
                if BACKSLASH in expr:
                    bad.append((tok.start[0], "backslash inside f-string expression", expr))
                i += 1
                continue
            if depth:
                expr += c
            i += 1

    if bad:
        print(f"FAIL {path.name}: {len(bad)} Python 3.11 f-string violation(s)")
        for line, why, snippet in bad:
            print(f"  line {line}: {why}")
            print(f"      {snippet[:70]!r}")
        return 1
    print(f"OK   {path.name}: no PEP 701-only f-string syntax")
    return 0


def check_undefined_names(path) -> int:
    """Names read but never bound anywhere that could bind them.

    This exists because of a bug that shipped: a CSS rule interpolated {BG}
    when the constant is called SURFACE. It parses — an f-string placeholder is
    valid syntax whatever is inside it — so the grammar check above passed, the
    deploy succeeded, and the app died at load with NameError on a line no test
    had ever executed. Parsing is not running.

    Deliberately conservative: a name is accepted if ANY scope in the file could
    plausibly bind it, so nested and conditional definitions do not produce
    noise. It is here to catch typos and renames, not to be a type checker.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    bound: set[str] = set(dir(builtins))
    used: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (bound.add(node.id) if isinstance(node.ctx, (ast.Store, ast.Del))
             else used.append((node.id, node.lineno)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            args = getattr(node, "args", None)
            if args:
                for a in (args.posonlyargs + args.args + args.kwonlyargs
                          + ([args.vararg] if args.vararg else [])
                          + ([args.kwarg] if args.kwarg else [])):
                    bound.add(a.arg)
        elif isinstance(node, ast.Lambda):
            a = node.args
            for x in (a.posonlyargs + a.args + a.kwonlyargs
                      + ([a.vararg] if a.vararg else [])
                      + ([a.kwarg] if a.kwarg else [])):
                bound.add(x.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            bound.update(node.names)

    missing = sorted({(n, ln) for n, ln in used if n not in bound})
    if missing:
        print(f"FAIL {path.name}: {len(missing)} undefined name(s)")
        for n, ln in missing[:20]:
            print(f"  line {ln}: {n!r} is read but never assigned, imported "
                  f"or defined anywhere in the file")
        return 1
    print(f"OK   {path.name}: no undefined names")
    return 0


if __name__ == "__main__":
    rc = 0
    for arg in sys.argv[1:]:
        rc |= check(Path(arg))
        rc |= check_undefined_names(Path(arg))
    sys.exit(rc)
