#!/usr/bin/env python3
"""Enforce the repository rules listed in docs/07 §2.

Usage: ``python scripts/check_rules.py [root]`` (default ``src/caracal``).
Prints ``path:line: rule: excerpt`` for every hit and exits 1 if any rule is broken.
Only ``.py`` files under the given root are inspected, so the checker never flags itself
or the tests unless they are explicitly passed as the root.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHITECTURE_DOC = REPO_ROOT / "docs" / "01-architecture.md"
DEFAULT_ROOT = "src/caracal"
MAX_EXCERPT = 160


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    rule: str
    excerpt: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.excerpt}"


@dataclass(frozen=True)
class LineRule:
    name: str
    pattern: re.Pattern[str]
    allow_config: bool = False


# Only comment/text markers and string-literal rules run on raw lines; every code rule is
# decided on the AST (docs/07 §2) so aliases and multi-line calls cannot slip through.
LINE_RULES: tuple[LineRule, ...] = (
    LineRule("todo-marker", re.compile(r"\b(?:TODO|FIXME|XXX)\b")),
    LineRule("stub-pass", re.compile(r"\bpass\s+#\s*stub\b")),
    LineRule("redis-prefix-literal", re.compile(r"[\"']caracal:"), allow_config=True),
)

_OS_ENVIRONMENT_NAMES = frozenset({"environ", "getenv", "putenv"})
_FORBIDDEN_IMPORT_RULES = {
    "requests": "requests-import",
    "arq": "queue-import",
    "rq": "queue-import",
    "celery": "queue-import",
}
_REDIS_FLUSH_NAMES = frozenset({"flushdb", "flushall"})


def _is_string_literal(node: ast.expr) -> bool:
    """True for a plain string constant or an f-string."""
    if isinstance(node, ast.JoinedStr):
        return True
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _is_true_constant(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


class RuleVisitor(ast.NodeVisitor):
    """Report code-rule violations on the AST, resolving local import aliases."""

    def __init__(self, path: Path, text: str, *, is_config: bool) -> None:
        self.path = path
        self.is_config = is_config
        self.lines = text.splitlines()
        self.aliases: dict[str, str] = {}
        self.violations: dict[tuple[int, str], Violation] = {}

    def collect_aliases(self, tree: ast.AST) -> None:
        """Map every local import name to its dotted target (``getenv -> os.getenv``)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    target = alias.name if alias.asname else alias.name.split(".")[0]
                    self.aliases[local] = target
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local = alias.asname or alias.name
                    self.aliases[local] = f"{module}.{alias.name}" if module else alias.name

    def add(self, node: ast.AST, rule: str) -> None:
        line = getattr(node, "lineno", 1)
        key = (line, rule)
        if key in self.violations:
            return
        excerpt = self.lines[line - 1].strip()[:MAX_EXCERPT] if 0 < line <= len(self.lines) else ""
        self.violations[key] = Violation(self.path, line, rule, excerpt)

    def resolve(self, node: ast.expr) -> str | None:
        """Return the canonical dotted name for a name/attribute expression."""
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value)
            return f"{base}.{node.attr}" if base is not None else None
        return None

    def _environment(self, node: ast.AST, rule: str) -> None:
        if not self.is_config:
            self.add(node, rule)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            rule = _FORBIDDEN_IMPORT_RULES.get(alias.name.split(".")[0])
            if rule is not None:
                self.add(node, rule)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        rule = _FORBIDDEN_IMPORT_RULES.get(module.split(".")[0])
        if rule is not None:
            self.add(node, rule)
        if module == "os":
            for alias in node.names:
                if alias.name in _OS_ENVIRONMENT_NAMES:
                    self._environment(node, "os-environ")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        canonical = self.resolve(node)
        if canonical in {"os.environ", "os.getenv", "os.putenv"}:
            self._environment(node, "os-environ")
        if node.attr == "NotImplementedError":
            self.add(node, "not-implemented")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "NotImplementedError":
            self.add(node, "not-implemented")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        canonical = self.resolve(node.func)
        if canonical is not None:
            self._check_call(node, canonical)
        self.generic_visit(node)

    def _check_call(self, node: ast.Call, canonical: str) -> None:
        if canonical in {"print", "builtins.print"}:
            self.add(node, "print-call")
        elif canonical in {"eval", "builtins.eval"}:
            self.add(node, "eval-call")
        elif canonical in {"exec", "builtins.exec"}:
            self.add(node, "exec-call")
        elif canonical in {"pickle.load", "pickle.loads"}:
            self.add(node, "pickle-load")
        elif canonical in {"os.getenv", "os.putenv"}:
            self._environment(node, "os-environ")
        elif canonical == "os.system":
            self.add(node, "os-system")
        elif canonical == "os.popen":
            self.add(node, "os-popen")
        elif canonical == "asyncio.create_subprocess_shell":
            self.add(node, "async-subprocess-shell")
        elif canonical.startswith("subprocess."):
            if any(keyword.arg == "shell" and _is_true_constant(keyword.value) for keyword in node.keywords):
                self.add(node, "shell-true")
            if node.args and _is_string_literal(node.args[0]):
                self.add(node, "subprocess-string-arg")

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in _REDIS_FLUSH_NAMES:
                self.add(node, "redis-flush")
            if node.func.attr == "keys":
                self.add(node, "redis-keys")
        elif canonical in _REDIS_FLUSH_NAMES:
            self.add(node, "redis-flush")


_BRACE_RE = re.compile(r"\{([^{}]*)\}")
_GLOB_SHAPE = re.compile(r"[A-Za-z0-9_./{},\-*]+")


@dataclass
class Layout:
    """The modules and directories that docs/01 §3 allows under ``src/caracal/``."""

    exact: set[str] = field(default_factory=set)
    directories: set[str] = field(default_factory=set)
    globs: list[str] = field(default_factory=list)

    def matches(self, relative: str) -> bool:
        if relative in self.exact:
            return True
        if any(relative.startswith(f"{directory}/") for directory in self.directories):
            return True
        return any(fnmatch.fnmatchcase(relative, pattern) for pattern in self.globs)

    def has_module_below(self, directory: str) -> bool:
        """True if the listing contains at least one module inside ``directory``."""
        prefix = f"{directory}/" if directory else ""
        if any(path.startswith(prefix) for path in self.exact):
            return True
        for listed in self.directories:
            if listed == directory or listed.startswith(prefix) or directory.startswith(f"{listed}/"):
                return True
        return any(pattern.startswith(prefix) for pattern in self.globs)


def _expand_braces(token: str) -> list[str]:
    match = _BRACE_RE.search(token)
    if match is None:
        return [token]
    expanded: list[str] = []
    for alternative in match.group(1).split(","):
        if not alternative:
            continue
        expanded.extend(_expand_braces(f"{token[: match.start()]}{alternative}{token[match.end() :]}"))
    return expanded or [token]


def _is_path_token(token: str) -> bool:
    if token.endswith(".py") or token.endswith("/"):
        return True
    # A description token can contain a "*" too (e.g. "key(*parts),"); require a path shape.
    return "*" in token and _GLOB_SHAPE.fullmatch(token) is not None


def _register(layout: Layout, token: str) -> None:
    if not _is_path_token(token):
        return
    clean = token.lstrip("./")
    if not clean:
        return
    if clean.endswith("/**"):
        layout.directories.add(clean[: -len("/**")])
    elif clean.endswith("/"):
        layout.directories.add(clean.rstrip("/"))
    elif "*" in clean:
        layout.globs.append(clean)
    else:
        layout.exact.add(clean)


def parse_layout(doc: Path = ARCHITECTURE_DOC) -> Layout:
    """Parse the ``src/caracal/`` entries of the repository-layout code block."""
    lines = doc.read_text(encoding="utf-8").splitlines()
    heading = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith("## 3.")),
        None,
    )
    if heading is None:
        raise SystemExit(f"check_rules: cannot find '## 3.' in {doc}")
    fence = next(
        (index for index in range(heading + 1, len(lines)) if lines[index].strip().startswith("```")),
        None,
    )
    if fence is None:
        raise SystemExit(f"check_rules: cannot find the layout code block in {doc}")

    layout = Layout()
    in_src = False
    for line in lines[fence + 1 :]:
        if line.strip().startswith("```"):
            break
        if not in_src:
            if line.strip() == "src/caracal/":
                in_src = True
            continue
        if line.strip() and not line[0].isspace():
            break
        for token in line.split():
            for candidate in _expand_braces(token):
                _register(layout, candidate)
    return layout


def _is_config_module(path: Path, root: Path) -> bool:
    return path.relative_to(root) == Path("config.py")


def _excerpt(line: str) -> str:
    return line.strip()[:MAX_EXCERPT]


def check_file(path: Path, root: Path, layout: Layout) -> list[Violation]:
    """Return every rule violation in one Python file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    is_config = _is_config_module(path, root)
    violations: list[Violation] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for rule in LINE_RULES:
            if rule.allow_config and is_config:
                continue
            if rule.pattern.search(line):
                violations.append(Violation(path, number, rule.name, _excerpt(line)))

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        line = exc.lineno or 1
        excerpt = _excerpt(text.splitlines()[line - 1]) if 0 < line <= len(text.splitlines()) else ""
        violations.append(Violation(path, line, "syntax-error", excerpt))
    else:
        visitor = RuleVisitor(path, text, is_config=is_config)
        visitor.collect_aliases(tree)
        visitor.visit(tree)
        violations.extend(visitor.violations.values())

    relative = path.relative_to(root).as_posix()
    if not layout.matches(relative):
        name = path.name
        parent = relative.rsplit("/", 1)[0] if "/" in relative else ""
        if not (name == "__init__.py" and layout.has_module_below(parent)):
            violations.append(Violation(path, 1, "unlisted-module", relative))
    return violations


def check_root(root: Path) -> list[Violation]:
    """Check every ``.py`` file under ``root``."""
    layout = parse_layout()
    violations: list[Violation] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        violations.extend(check_file(path, root, layout))
    return sorted(violations, key=lambda item: (str(item.path), item.line, item.rule))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_rules.py",
        description="Enforce the Caracal repository rules (docs/07 §2).",
    )
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT, help="directory to scan")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        sys.stderr.write(f"check_rules: not a directory: {root}\n")
        return 2
    root = root.resolve()
    violations = check_root(root)
    for violation in violations:
        print(violation.render())
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
