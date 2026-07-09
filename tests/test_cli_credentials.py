from __future__ import annotations

import os
import stat

import pytest

from cli import credentials


def test_save_token_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKVISIONZ_CONFIG_DIR", str(tmp_path))
    credentials.save_token("svz_abc123")
    creds = credentials.load_credentials()
    assert creds is not None and creds.token == "svz_abc123"


def test_save_token_rejects_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKVISIONZ_CONFIG_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        credentials.save_token("   ")


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions only")
def test_save_token_is_owner_only(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKVISIONZ_CONFIG_DIR", str(tmp_path))
    credentials.save_token("svz_secret")
    mode = stat.S_IMODE(os.stat(credentials.credentials_path()).st_mode)
    assert mode == 0o600
