#!/usr/bin/env python3
"""Keep each test program to one native compile (chelis#2928).

`chelis build --output DIR` compiles the generated sources and publishes an
executable or a static library; `--emit-c` writes the sources without
compiling them (spec/08-backends.md §7). A test that runs the default build
and then compiles or links the generated C itself compiles every program
twice. A test that only runs the program uses the executable the build
publishes; a test that compiles the generated sources itself passes
`--emit-c`.

The guard reads every Rust file under `crates/` and reports each function
that both runs a default build and compiles C itself:

- A function runs a default build when a `"build"` string literal opens an
  argument list in it (`["build", ...]`, `.arg("build")`, or the first
  argument of a call to a same-file function that does not name
  `"--emit-c"`) and the function itself does not name `"--emit-c"`.
- A function compiles C when it calls `link_generated`, or a `Type::new`
  whose argument names a C compiler: a literal such as `"cc"`, or a name such
  as `compiler`, `c_compiler` or `cc`. A compile whose chained arguments name
  a static library other than the runtime archive links a driver against a
  published library and does not count.

A function inherits both facts from the same-file functions it calls, except
that a function naming `"--emit-c"` inherits no default build, so a helper
that builds and a helper that links are reported where they meet. The guard
follows no values: a command name held in a variable, a build or compile
helper defined in another file (other than `link_generated`), a compiler
reached under another name, and a function that mixes a source-only build
with a default one are not seen.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

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
    calls: dict[int, set[str]] = {}
    for function in found:
        callees = calls.setdefault(id(function), set())
        for index in function.own:
            token = tokens[index]
            previous = tokens[index - 1]
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if (
                token.kind == "ident"
                and token.text in by_name
                and following is not None
                and following.text == "("
                and previous.text not in {"fn", "."}
            ):
                callees.add(token.text)
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
                helpers = by_name.get(opener, []) if tokens[index - 3].text not in {"fn", "."} else []
                if opener == "arg" or (helpers and not any(h.has_emit_c for h in helpers)):
                    function.build_line = token.line
    # Inherit both facts from same-file callees until nothing changes; a
    # function that names `--emit-c` itself inherits no default build.
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


def rust_files(root: Path) -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "crates"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return sorted(path for path in listed if path.endswith(".rs") and (root / path).is_file())


def check(root: Path = ROOT, paths: Iterable[str] | None = None) -> list[str]:
    errors: list[str] = []
    for path in rust_files(root) if paths is None else paths:
        source = (root / path).read_text(encoding="utf-8")
        if '"build"' not in source:
            continue
        try:
            flagged = analyze(source)
        except ValueError as error:
            errors.append(f"{path}: cannot scan: {error}")
            continue
        for function in flagged:
            errors.append(
                f"{path}:{function.line}: fn {function.name} runs a default `chelis build` "
                f"(line {function.build_line}) and compiles C itself (line "
                f"{function.compile_line}); run the published executable, or pass "
                f"`--emit-c` when the test compiles the generated sources"
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
