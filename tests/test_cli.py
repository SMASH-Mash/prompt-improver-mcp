import json

import pytest

from prompt_improver import cli


@pytest.fixture(autouse=True)
def no_api_credentials(monkeypatch):
    monkeypatch.setattr(cli, "anthropic_credentials_available", lambda: False)


def test_cli_prints_draft_and_techniques(capsys):
    assert cli.main(["How many red cubes are behind the sphere?", "--image", "--target", "standard"]) == 0
    out = capsys.readouterr().out
    assert "<SUMMARY>" in out and "backend: host" in out and "arXiv:2411.10440" in out


def test_cli_json_with_history_file(tmp_path, capsys):
    history = tmp_path / "history.json"
    history.write_text(json.dumps([
        {"role": "user", "content": "Write a regex for ISO dates."},
        {"role": "assistant", "content": r"\d{4}-\d{2}-\d{2}"},
    ]), encoding="utf-8")
    assert cli.main(["no, it still accepts month 13", "--history", str(history), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "failure_feedback" in [t["id"] for t in data["techniques"]]


def test_cli_reports_bad_history(tmp_path, capsys):
    bad = tmp_path / "history.json"
    bad.write_text('[{"role": "robot", "content": "hi"}]', encoding="utf-8")
    assert cli.main(["fix it", "--history", str(bad)]) == 2
    assert "error" in capsys.readouterr().err
