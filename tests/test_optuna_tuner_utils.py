from pathlib import Path

import pytest


def test_default_storage_uri_is_sqlite(tmp_path: Path):
    from src.tuning.optuna_tuner import default_storage_uri

    uri = default_storage_uri(tmp_path, "study")
    assert uri.startswith("sqlite:///")
    assert uri.endswith("/study.db") or uri.endswith("\\study.db") or uri.endswith("study.db")


def test_resolve_storage_uri_uses_default_when_none(tmp_path: Path):
    from src.tuning.optuna_tuner import resolve_storage_uri

    uri = resolve_storage_uri(None, tmp_path, "study")
    assert uri.startswith("sqlite:///")
    assert "optuna" in uri


def test_require_optuna_raises_helpful_error_when_missing(monkeypatch):
    import src.tuning.optuna_tuner as tuner

    def _boom(_name: str):
        raise ImportError("no optuna")

    monkeypatch.setattr(tuner, "import_module", _boom)
    with pytest.raises(RuntimeError, match="Optuna is not installed"):
        tuner.require_optuna()

