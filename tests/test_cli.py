"""CLI: ingest / save / info / ask round-trips (LLM-free paths)."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cli

DOC = "# Outage\n\n## Cause\nThe fire caused a power loss. The power loss disrupted the hospital.\n"


def _write(tmp, name, text):
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def test_ingest_save_and_info(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        doc = _write(tmp, "doc.md", DOC)
        pkl = os.path.join(tmp, "g.pkl")

        rc = cli.main(["ingest", doc, "--save", pkl])
        assert rc == 0
        assert os.path.exists(pkl)
        out = capsys.readouterr().out
        assert "Ingested" in out and "edges" in out

        rc = cli.main(["info", pkl])
        assert rc == 0
        info = capsys.readouterr().out
        assert "nodes:" in info and "causal edges:" in info
        assert "fire" in info  # sample edge rendered


def test_missing_file_returns_clean_error(capsys):
    rc = cli.main(["ingest", "does_not_exist_12345.md"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower()


def test_parser_requires_subcommand():
    import pytest
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])
