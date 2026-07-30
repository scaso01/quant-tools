"""Regression tests for argument handling in `quant-tools web`.

_cmd_web took an args list and never used it, so every option a user passed was
dropped. The README's own testing instructions say

    QUANT_TOOLS_OFFLINE=1 quant-tools web --server.port 8599

which silently started on 8501 instead, and --server.headless could not be
turned on at all, so the command always opened a browser tab.
"""

from __future__ import annotations

from unittest.mock import patch

from quant_tools.cli import _cmd_web


def _argv(args):
    with patch("subprocess.run") as run:
        _cmd_web(args)
        return run.call_args[0][0]


def _value_of(argv, flag):
    return argv[argv.index(flag) + 1]


class TestDefaults:
    def test_runs_the_dashboard_app(self):
        argv = _argv([])
        assert argv[1:4] == ["-m", "streamlit", "run"]
        assert argv[4].endswith("app.py")

    def test_theme_is_pinned(self):
        """Streamlit reads .streamlit/config.toml relative to the cwd."""
        argv = _argv([])
        assert _value_of(argv, "--theme.base") == "dark"
        assert _value_of(argv, "--theme.backgroundColor") == "#0e1117"

    def test_headless_defaults_off(self):
        assert _value_of(_argv([]), "--server.headless") == "false"


class TestUserArgs:
    def test_port_is_forwarded(self):
        """The bug: this used to be dropped and the app started on 8501."""
        argv = _argv(["--server.port", "8599"])
        assert _value_of(argv, "--server.port") == "8599"

    def test_override_replaces_the_default(self):
        argv = _argv(["--server.headless", "true"])
        assert argv.count("--server.headless") == 1
        assert _value_of(argv, "--server.headless") == "true"

    def test_equals_form_also_overrides(self):
        argv = _argv(["--server.headless=true"])
        assert argv.count("--server.headless") == 0
        assert "--server.headless=true" in argv

    def test_untouched_defaults_survive_an_override(self):
        argv = _argv(["--server.headless", "true"])
        assert _value_of(argv, "--theme.base") == "dark"
        assert _value_of(argv, "--client.toolbarMode") == "minimal"

    def test_unknown_options_pass_through(self):
        argv = _argv(["--server.enableCORS", "false"])
        assert _value_of(argv, "--server.enableCORS") == "false"
