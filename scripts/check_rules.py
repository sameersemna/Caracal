#!/usr/bin/env python3
"""Enforce the repository rules listed in docs/07 §2.

Usage: ``python scripts/check_rules.py [root]`` (default ``src/caracal``).
Prints ``path:line: rule: excerpt`` for every hit and exits 1 if any rule is broken.
Only ``.py`` files under the given root are inspected, so the checker never flags itself
or the tests unless they are explicitly passed as the root.
"""

from __future__ import annotations

import argparse
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


LINE_RULES: tuple[LineRule, ...] = (
    LineRule("os-environ", re.compile(r"\bos\.(?:environ|getenv)\b"), allow_config=True),
    LineRule("print-call", re.compile(r"\bprint\s*\(")),
    LineRule("shell-true", re.compile(r"\bshell\s*=\s*True\b")),
    LineRule("os-system", re.compile(r"\bos\.system\s*\(")),
    LineRule("subprocess-string-arg", re.compile(r"\bsubprocess\.(?:call|run)\s*\(\s*[rbfuRBFU]{0,2}[\"']")),
    LineRule("eval-call", re.compile(r"(?<![\w.])eval\s*\(")),
    LineRule("exec-call", re.compile(r"(?<![\w.])exec\s*\(")),
    LineRule("pickle-load", re.compile(r"\bpickle\.loads?\s*\(")),
    LineRule("todo-marker", re.compile(r"\b(?:TODO|FIXME|XXX)\b")),
    LineRule("not-implemented", re.compile(r"\bNotImplementedError\b")),
    LineRule("stub-pass", re.compile(r"\bpass\s+#\s*stub\b")),
    LineRule("requests-import", re.compile(r"^\s*(?:import|from)\s+requests\b")),
    LineRule("queue-import", re.compile(r"^\s*(?:import|from)\s+(?:arq|rq|celery)\b")),
    LineRule("redis-flush", re.compile(r"\b(?:flushdb|flushall)\b")),
    LineRule("redis-keys", re.compile(r"\.keys\s*\(")),
    LineRule("redis-prefix-literal", re.compile(r"[\"']caracal:"), allow_config=True),
)

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
