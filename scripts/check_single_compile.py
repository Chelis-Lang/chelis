#!/usr/bin/env python3
"""Keep each test program to one native compile (chelis#2928).

`chelis build --output DIR` compiles the generated sources and publishes an
executable or a static library; `--emit-c` writes the sources without
compiling them (spec/08-backends.md §7). A test or harness that runs the
default build and then compiles or links the generated C itself compiles
every program twice. A caller that only runs the program uses the executable
the build publishes; a caller that compiles the generated sources itself
passes `--emit-c`.

The guard reads every Rust file under `crates/` and every Python file under
`scripts/` and `.github/scripts/`, and reports each function that both runs a
default build and compiles C itself. In Rust:

- A function runs a default build when a `"build"` string literal opens an
  argument list in it (`["build", ...]`, `.arg("build")`, or the first
  argument of a call to a same-file function or method that does not name
  `"--emit-c"`) and the function itself does not name `"--emit-c"`.
- A function compiles C when it calls `link_generated`, or a `Type::new`
  whose argument names a C compiler: a literal such as `"cc"`, or a name such
  as `compiler`, `c_compiler` or `cc`. A compile whose chained arguments name
  a static library other than the runtime archive links a driver against a
  published library and does not count.

In Python, read with the standard `ast` module:

- A function (module-level statements count as one) runs a default build
  when a list or tuple literal in it holds the string `"build"` first or
  right after a program that is not a string literal (`[chelis, "build",
  ...]`), and the function names no `"--emit-c"`.
- It compiles C when a list or tuple literal starts with a compiler: a name
  or attribute such as `compiler`, `toolchain.compiler` or `cc` (also inside
  `str(...)`), or a literal such as `"cc"`. A command that names a static
  library other than the runtime archive links a driver against a published
  library and does not count.

In both languages a function inherits both facts from the same-file
functions and methods it calls, as `f(...)` or `x.f(...)`, except that a
function naming `"--emit-c"` inherits no default build, so a helper that
builds and a helper that links are reported where they meet. Calls resolve by
name alone: a method of any type resolves to every same-file function of that
name. The guard follows no values: a command name held in a variable, a
`"build"` placed elsewhere in a list (Rust) or after a literal program
(Python, where `["cargo", "build"]` is another tool), a Python `"build"`
passed as a call argument rather than in a list or tuple, a build or compile
helper defined in another file (other than `link_generated`), a compiler
reached under another name, and a function that mixes a source-only build
with a default one are not seen.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Union

ROOT = Path(__file__).resolve().parents[1]
EMIT_C = "--emit-c"
COMPILE_HELPER = "link_generated"
COMPILER_LITERALS = frozenset(
    {"cc", "gcc", "clang", "c++", "g++", "clang++", "hipcc", "xcrun"}
)
COMPILER_NAMES = frozenset({"cc", "cxx", "gcc", "clang", "hipcc"})
RUNTIME_ARCHIVE = "libchelis_runtime.a"
MODULE_LIBRARY = re.compile(r"(?:^|/)lib[^/]+\.a$")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUMBER = re.compile(r"[0-9][0-9A-Za-z_]*")
RAW_STRING = re.compile(r'(?:b|c)?r(#*)"')
PYTHON_ROOTS = ("scripts", ".github/scripts")
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class Token:
    kind: str  # "ident", "str", "punct" or "other"
    text: str
    line: int


def tokenize(source: str) -> list[Token]:
    """Rust tokens, comments dropped; a string literal keeps its source text."""
    tokens: list[Token] = []
    i, line, n = 0, 1, len(source)

    def advance(end: int) -> None:
        nonlocal i, line
        line += source.count("\n", i, end)
        i = end

    while i < n:
        char = source[i]
        if char in " \t\r\n":
            advance(i + 1)
        elif source.startswith("//", i):
            end = source.find("\n", i)
            advance(n if end == -1 else end)
        elif source.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if source.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif source.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            advance(j)
        elif raw := RAW_STRING.match(source, i):
            terminator = '"' + raw.group(1)
            end = source.find(terminator, raw.end())
            if end == -1:
                raise ValueError(f"unterminated raw string at line {line}")
            tokens.append(Token("str", source[raw.end() : end], line))
            advance(end + len(terminator))
        elif char == '"' or (char in "bc" and source.startswith('"', i + 1)):
            start = i + (char != '"') + 1
            j = start
            while j < n and source[j] != '"':
                j += 2 if source[j] == "\\" else 1
            if j >= n:
                raise ValueError(f"unterminated string at line {line}")
            tokens.append(Token("str", source[start:j], line))
            advance(j + 1)
        elif char == "'":
            # A character literal closes within a short escape; a lifetime or
            # label does not close at all.
            if source.startswith("\\", i + 1):
                end = source.find("'", i + 3)
                if end == -1:
                    raise ValueError(f"unterminated character literal at line {line}")
                advance(end + 1)
            elif i + 2 < n and source[i + 2] == "'":
                advance(i + 3)
            else:
                tokens.append(Token("punct", char, line))
                advance(i + 1)
        elif match := IDENT.match(source, i):
            tokens.append(Token("ident", match.group(), line))
            advance(match.end())
        elif char.isdigit():
            match = NUMBER.match(source, i)
            tokens.append(Token("other", match.group(), line))
            advance(match.end())
        else:
            tokens.append(Token("punct", char, line))
            advance(i + 1)
    return tokens


@dataclass
class Function:
    name: str
    line: int
    body: range  # token indices of `{` .. `}`
    own: list[int]  # body token indices outside nested functions
    has_emit_c: bool = False
    build_line: int | None = None
    compile_line: int | None = None


def matching(tokens: Sequence[Token]) -> dict[int, int]:
    """Map each `(`, `[` and `{` index to its closing index."""
    pairs: dict[int, int] = {}
    stack: list[int] = []
    closers = {")": "(", "]": "[", "}": "{"}
    for index, token in enumerate(tokens):
        if token.kind != "punct":
            continue
        if token.text in "([{":
            stack.append(index)
        elif token.text in closers:
            if not stack or tokens[stack[-1]].text != closers[token.text]:
                raise ValueError(f"unbalanced {token.text!r} at line {token.line}")
            pairs[stack.pop()] = index
    if stack:
        raise ValueError(f"unclosed {tokens[stack[-1]].text!r} at line {tokens[stack[-1]].line}")
    return pairs


def functions(tokens: Sequence[Token], pairs: dict[int, int]) -> list[Function]:
    found: list[Function] = []
    for index, token in enumerate(tokens[:-1]):
        name = tokens[index + 1]
        if token.kind != "ident" or token.text != "fn" or name.kind != "ident":
            continue
        cursor = index + 2
        while cursor < len(tokens):
            text = tokens[cursor].text
            if tokens[cursor].kind == "punct" and text in "([":
                cursor = pairs[cursor] + 1
            elif tokens[cursor].kind == "punct" and text in "{;":
                break
            else:
                cursor += 1
        if cursor < len(tokens) and tokens[cursor].text == "{":
            body = range(cursor, pairs[cursor] + 1)
            found.append(Function(name.text, name.line, body, []))
    for function in found:
        nested = [
            other.body
            for other in found
            if other is not function and other.body.start in function.body
        ]
        function.own = [
            index
            for index in function.body
            if not any(index in inner for inner in nested)
        ]
    return found


def is_compile(tokens: Sequence[Token], pairs: dict[int, int], index: int) -> bool:
    """Whether the call at `index` compiles C: `link_generated(...)` or a compiler `Type::new(...)`."""
    token = tokens[index]
    following = tokens[index + 1] if index + 1 < len(tokens) else None
    if following is None or following.text != "(" or following.kind != "punct":
        return False
    previous = tokens[index - 1] if index else None
    if token.text == COMPILE_HELPER:
        return previous is None or previous.text != "fn"
    if not (
        token.text == "new"
        and index >= 3
        and tokens[index - 1].text == ":"
        and tokens[index - 2].text == ":"
        and tokens[index - 3].text[:1].isupper()
    ):
        return False
    argument = tokens[index + 2 : pairs[index + 1]]
    names_compiler = any(
        (part.kind == "str" and part.text in COMPILER_LITERALS)
        or (
            part.kind == "ident"
            and (
                part.text in COMPILER_NAMES
                or part.text == "compiler"
                or part.text.endswith("_compiler")
            )
        )
        for part in argument
    )
    if not names_compiler:
        return False
    # The command chained onto this constructor, up to its statement's end.
    end = pairs[index + 1]
    while end + 1 < len(tokens) and tokens[end + 1].text not in ";{}":
        end += 1
        if tokens[end].text in "([":
            end = pairs[end]
    links_library = any(
        part.kind == "str" and MODULE_LIBRARY.search(part.text) and not part.text.endswith(RUNTIME_ARCHIVE)
        for part in tokens[index : end + 1]
    )
    return not links_library


def analyze(source: str) -> list[Function]:
    """The functions of `source` that run a default build and compile C."""
    tokens = tokenize(source)
    pairs = matching(tokens)
    found = functions(tokens, pairs)
    by_name: dict[str, list[Function]] = {}
    for function in found:
        by_name.setdefault(function.name, []).append(function)
        function.has_emit_c = any(
            tokens[index].kind == "str" and tokens[index].text == EMIT_C
            for index in function.own
        )
    # Callees in first-call order, so the cited lines do not depend on hashing.
    calls: dict[int, dict[str, None]] = {}
    for function in found:
        callees = calls.setdefault(id(function), {})
        for index in function.own:
            token = tokens[index]
            previous = tokens[index - 1]
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if (
                token.kind == "ident"
                and token.text in by_name
                and following is not None
                and following.text == "("
                and previous.text != "fn"
            ):
                callees.setdefault(token.text)
            if (
                token.kind == "ident"
                and function.compile_line is None
                and is_compile(tokens, pairs, index)
            ):
                function.compile_line = token.line
            if (
                token.kind != "str"
                or token.text != "build"
                or function.has_emit_c
                or function.build_line is not None
            ):
                continue
            if previous.text == "[":
                function.build_line = token.line
            elif previous.text == "(" and tokens[index - 2].kind == "ident":
                opener = tokens[index - 2].text
                helpers = by_name.get(opener, []) if tokens[index - 3].text != "fn" else []
                if opener == "arg" or (helpers and not any(h.has_emit_c for h in helpers)):
                    function.build_line = token.line
    return inherit(found, calls, by_name)


@dataclass
class PyFunction:
    name: str
    line: int
    node: ast.AST
    has_emit_c: bool = False
    build_line: int | None = None
    compile_line: int | None = None


Scanned = Union[Function, PyFunction]


def inherit(
    found: Sequence[Scanned],
    calls: dict[int, dict[str, None]],
    by_name: dict[str, list],
) -> list:
    """Spread both facts from same-file callees until nothing changes; return the flagged.

    A function that names `--emit-c` itself inherits no default build.
    """
    changed = True
    while changed:
        changed = False
        for function in found:
            for callee in (c for name in calls[id(function)] for c in by_name[name]):
                if callee is function:
                    continue
                if (
                    function.build_line is None
                    and not function.has_emit_c
                    and callee.build_line is not None
                ):
                    function.build_line = callee.build_line
                    changed = True
                if function.compile_line is None and callee.compile_line is not None:
                    function.compile_line = callee.compile_line
                    changed = True
    return [
        function
        for function in found
        if function.build_line is not None and function.compile_line is not None
    ]


def own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """The nodes under `node`, outside nested function definitions."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, FUNCTION_NODES):
            continue
        yield child
        yield from own_nodes(child)


def is_text(node: ast.AST, text: str | None = None) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and (text is None or node.value == text)
    )


def names_compiler(node: ast.AST) -> bool:
    """Whether a command's first element is a C compiler."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and len(node.args) == 1
    ):
        node = node.args[0]
    name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
    if name is not None:
        return name in COMPILER_NAMES or name == "compiler" or name.endswith("_compiler")
    return is_text(node) and node.value in COMPILER_LITERALS


def py_compile_line(command: ast.List | ast.Tuple) -> int | None:
    if not command.elts or not names_compiler(command.elts[0]):
        return None
    links_library = any(
        is_text(part)
        and MODULE_LIBRARY.search(part.value)
        and not part.value.endswith(RUNTIME_ARCHIVE)
        for part in ast.walk(command)
    )
    return None if links_library else command.lineno


def py_build_line(command: ast.List | ast.Tuple) -> int | None:
    for index, element in enumerate(command.elts):
        if is_text(element, "build") and (index == 0 or not is_text(command.elts[index - 1])):
            return element.lineno
    return None


def analyze_python(source: str) -> list[PyFunction]:
    """The functions of Python `source` that run a default build and compile C."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"not parseable Python: {error.msg} at line {error.lineno}") from error
    found = [PyFunction("<module>", 1, tree)] + [
        PyFunction(node.name, node.lineno, node)
        for node in ast.walk(tree)
        if isinstance(node, FUNCTION_NODES)
    ]
    found.sort(key=lambda function: function.line)
    by_name: dict[str, list[PyFunction]] = {}
    for function in found:
        by_name.setdefault(function.name, []).append(function)
    calls: dict[int, dict[str, None]] = {}
    for function in found:
        nodes = sorted(
            (node for node in own_nodes(function.node) if hasattr(node, "lineno")),
            key=lambda node: (node.lineno, node.col_offset),
        )
        function.has_emit_c = any(is_text(node, EMIT_C) for node in nodes)
        callees = calls.setdefault(id(function), {})
        for node in nodes:
            if isinstance(node, ast.Call):
                callee = node.func
                name = callee.id if isinstance(callee, ast.Name) else callee.attr if isinstance(callee, ast.Attribute) else None
                if name in by_name:
                    callees.setdefault(name)
            if not isinstance(node, (ast.List, ast.Tuple)):
                continue
            if function.compile_line is None:
                function.compile_line = py_compile_line(node)
            if function.build_line is None and not function.has_emit_c:
                function.build_line = py_build_line(node)
    return inherit(found, calls, by_name)


def source_files(root: Path) -> list[str]:
    """The Rust files under `crates/` and the Python files under the script roots."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "crates", *PYTHON_ROOTS],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return sorted(
        path
        for path in listed
        if (
            (path.startswith("crates/") and path.endswith(".rs"))
            or (path.startswith(tuple(f"{prefix}/" for prefix in PYTHON_ROOTS)) and path.endswith(".py"))
        )
        and (root / path).is_file()
    )


def check(root: Path = ROOT, paths: Iterable[str] | None = None) -> list[str]:
    errors: list[str] = []
    for path in source_files(root) if paths is None else paths:
        source = (root / path).read_text(encoding="utf-8")
        python = path.endswith(".py")
        if '"build"' not in source and not (python and "'build'" in source):
            continue
        try:
            flagged: list[Scanned] = list(analyze_python(source) if python else analyze(source))
        except ValueError as error:
            errors.append(f"{path}: cannot scan: {error}")
            continue
        keyword = "def" if python else "fn"
        for function in flagged:
            errors.append(
                f"{path}:{function.line}: {keyword} {function.name} runs a default `chelis build` "
                f"(line {function.build_line}) and compiles C itself (line "
                f"{function.compile_line}); run the published executable, or pass "
                f"`--emit-c` when the caller compiles the generated sources"
            )
    return errors


def main() -> int:
    errors = check()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print("SINGLE COMPILE: FAIL", file=sys.stderr)
        return 1
    print("SINGLE COMPILE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
