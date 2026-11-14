"""The repository rules checker (docs/07 §2)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("caracal_check_rules", SCRIPTS / "check_rules.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_rules = _load_checker()


def _write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _hits(root: Path) -> list[object]:
    return check_rules.check_root(root)


@pytest.mark.parametrize(
    ("rule", "source", "line"),
    [
        ("os-environ", 'import os\nvalue = os.environ["X"]\n', 2),
        ("os-environ", 'import os\nvalue = os.getenv("X")\n', 2),
        ("print-call", 'print("hello")\n', 1),
        ("shell-true", 'import subprocess\nsubprocess.run(["ls"], shell=True)\n', 2),
        ("os-system", 'import os\nos.system("rm -rf /")\n', 2),
        ("subprocess-string-arg", 'import subprocess\nsubprocess.call("ls -l")\n', 2),
        ("subprocess-string-arg", 'import subprocess\nsubprocess.run("ls -l")\n', 2),
        ("eval-call", 'eval("1 + 1")\n', 1),
        ("exec-call", 'exec("x = 1")\n', 1),
        ("pickle-load", "import pickle\npickle.load(handle)\n", 2),
        ("todo-marker", "# TODO fix this\n", 1),
        ("todo-marker", 'x = "FIXME"\n', 1),
        ("todo-marker", "# XXX\n", 1),
        ("not-implemented", "raise NotImplementedError\n", 1),
        ("stub-pass", "def f():\n    pass  # stub\n", 2),
        ("requests-import", "import requests\n", 1),
        ("queue-import", "import celery\n", 1),
        ("queue-import", "from rq import Queue\n", 1),
        ("queue-import", "import arq\n", 1),
        ("redis-flush", "client.flushdb()\n", 1),
        ("redis-flush", "client.flushall()\n", 1),
        ("redis-keys", 'client.keys("*")\n', 1),
        ("redis-prefix-literal", 'key = "caracal:jobs"\n', 1),
    ],
)
def test_each_line_rule_is_reported_at_the_right_line(
    tmp_path: Path, rule: str, source: str, line: int
) -> None:
    _write(tmp_path, "cli.py", source)
    hits = _hits(tmp_path)
    assert len(hits) == 1
    assert hits[0].rule == rule
    assert hits[0].line == line


def test_clean_file_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "cli.py",
        'import subprocess\nsubprocess.run(["ls"], check=True)\n\n\ndef run() -> int:\n    return 1\n',
    )
    assert _hits(tmp_path) == []


def test_brace_lists_are_expanded() -> None:
    assert check_rules._expand_braces("analysis/{content,identity}.py") == [
        "analysis/content.py",
        "analysis/identity.py",
    ]


def test_double_star_and_trailing_slash_entries_cover_every_file_below() -> None:
    layout = check_rules.Layout()
    check_rules._register(layout, "jobs/**")
    check_rules._register(layout, "migrations/")
    assert layout.matches("jobs/a/b.py")
    assert layout.matches("migrations/versions/0001_init.py")
    assert not layout.matches("jobs.py")


def test_description_token_with_a_star_is_not_a_path() -> None:
    layout = check_rules.Layout()
    check_rules._register(layout, "key(*parts),")
    assert layout.globs == []
    assert layout.exact == set()


def test_unlisted_module_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "extra.py", "VALUE = 1\n")
    hits = _hits(tmp_path)
    assert [hit.rule for hit in hits] == ["unlisted-module"]
    assert hits[0].excerpt == "extra.py"
    assert hits[0].line == 1


def test_listed_module_passes(tmp_path: Path) -> None:
    _write(tmp_path, "api/routes.py", "VALUE = 1\n")
    assert _hits(tmp_path) == []


def test_listed_directory_passes_for_every_file_below_it(tmp_path: Path) -> None:
    _write(tmp_path, "migrations/env.py", "VALUE = 1\n")
    _write(tmp_path, "migrations/versions/0001_init.py", "VALUE = 1\n")
    assert _hits(tmp_path) == []


def test_init_is_allowed_in_a_directory_with_a_listed_module(tmp_path: Path) -> None:
    _write(tmp_path, "__init__.py", "")
    _write(tmp_path, "api/__init__.py", "")
    _write(tmp_path, "llm/prompts/__init__.py", "")
    assert _hits(tmp_path) == []


def test_init_is_rejected_in_a_directory_without_listed_modules(tmp_path: Path) -> None:
    _write(tmp_path, "randomdir/__init__.py", "")
    hits = _hits(tmp_path)
    assert [(hit.rule, hit.excerpt) for hit in hits] == [
        ("unlisted-module", "randomdir/__init__.py"),
    ]


def test_non_python_files_are_ignored(tmp_path: Path) -> None:
    _write(tmp_path, "llm/prompts/stage.md", "print('not python')\nTODO\n")
    assert _hits(tmp_path) == []


def test_config_module_is_exempt_from_environment_and_prefix_rules(tmp_path: Path) -> None:
    _write(tmp_path, "config.py", 'import os\nkey = "caracal:jobs"\nvalue = os.environ["X"]\n')
    assert _hits(tmp_path) == []


def test_redis_prefix_literal_is_reported_outside_config(tmp_path: Path) -> None:
    _write(tmp_path, "queue.py", 'PREFIX = "caracal:queues"\n')
    hits = _hits(tmp_path)
    assert [hit.rule for hit in hits] == ["redis-prefix-literal"]


def test_multiple_violations_are_all_reported(tmp_path: Path) -> None:
    _write(tmp_path, "cli.py", 'print("a")\n# FIXME later\n')
    hits = _hits(tmp_path)
    assert [(hit.rule, hit.line) for hit in hits] == [("print-call", 1), ("todo-marker", 2)]


def test_cli_reports_hits_and_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path, "extra.py", "VALUE = 1\n")
    assert check_rules.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "unlisted-module" in out
    assert f"{tmp_path / 'extra.py'}:1:" in out


def test_cli_exits_zero_for_a_clean_tree(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path, "cli.py", "VALUE = 1\n")
    assert check_rules.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out == ""


def test_cli_rejects_a_missing_root(tmp_path: Path) -> None:
    assert check_rules.main([str(tmp_path / "missing")]) == 2


def test_repository_source_is_clean() -> None:
    package_root = Path(check_rules.__file__).resolve().parents[1] / "src" / "caracal"
    assert _hits(package_root) == []
