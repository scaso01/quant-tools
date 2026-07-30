"""Contract tests for the top-level CLI dispatcher.

Nothing exercised `quant_tools/cli.py` itself, so commands could name a class or
a method that no longer existed and still look healthy: the import sits inside
the handler, so the failure only appears when someone runs that command. These
tests walk the dispatcher's own source and check that what it reaches for is
really there.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

from quant_tools import cli

CLI_SOURCE = Path(inspect.getfile(cli))


def _first_party_imports() -> list[tuple[str, str, int]]:
    """Every `from quant_tools.x import Y` written inside cli.py, with its line."""
    tree = ast.parse(CLI_SOURCE.read_text(encoding="utf-8"), filename=str(CLI_SOURCE))
    found: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("quant_tools"):
                for alias in node.names:
                    found.append((node.module, alias.name, node.lineno))
    return found


def test_cli_source_has_imports_to_check() -> None:
    """Guard the guard: an empty list would make the next test vacuously pass."""
    assert len(_first_party_imports()) > 20


@pytest.mark.parametrize(
    ("module_name", "attr", "lineno"),
    _first_party_imports(),
    ids=lambda v: str(v),
)
def test_cli_import_resolves(module_name: str, attr: str, lineno: int) -> None:
    """Each name the dispatcher imports exists in the module it imports from."""
    module = importlib.import_module(module_name)
    assert hasattr(module, attr), (
        f"{CLI_SOURCE.name}:{lineno} imports {attr!r} from {module_name!r}, "
        f"which does not define it"
    )


def _dispatch_table() -> set[str]:
    """The command names main() actually dispatches on."""
    tree = ast.parse(CLI_SOURCE.read_text(encoding="utf-8"), filename=str(CLI_SOURCE))
    main_fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    table = next(n for n in ast.walk(main_fn) if isinstance(n, ast.Dict))
    return {
        k.value for k in table.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }


def test_every_advertised_command_has_a_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every command named in the help text is dispatchable.

    The help text is the first thing a new user reads, so a command listed there
    with no handler is a broken promise on the very first run.
    """
    printed: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    cli._print_help()

    # Command rows are indented and two-column: "  scan    Scan universe...".
    # The example lines below them all start with "quant-tools".
    advertised = {
        line.split()[0]
        for line in printed
        if line.startswith("  ") and line.split() and not line.strip().startswith("quant-tools")
    }
    unhandled = advertised - _dispatch_table()
    assert not unhandled, f"help lists commands with no handler: {sorted(unhandled)}"


@pytest.mark.parametrize("command", ["dashboard", "web"])
def test_help_returns_without_launching(command: str, capsys: pytest.CaptureFixture) -> None:
    """--help must print and return.

    `dashboard` used to ignore its arguments and open the full-screen TUI, and
    `web` handed --help to streamlit, whose help text cannot even be encoded on a
    Windows console. Both left the user stuck.
    """
    handler = getattr(cli, f"_cmd_{command}")
    handler(["--help"])
    out = capsys.readouterr().out
    assert out.startswith(f"usage: quant-tools {command}")
