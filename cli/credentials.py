from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _config_dir() -> Path:
    root = os.environ.get("STOCKVISIONZ_CONFIG_DIR")
    if root:
        return Path(root).expanduser().resolve()
    if os.name == "nt":
        appdata = os.environ.get("APPDATA") or str(Path.home())
        return Path(appdata) / "stockvisionz"
    return Path.home() / ".config" / "stockvisionz"


def credentials_path() -> Path:
    return _config_dir() / "credentials.json"


def _credentials_path() -> Path:
    return credentials_path()


@dataclass(frozen=True)
class Credentials:
    token: str


def load_credentials() -> Credentials | None:
    path = _credentials_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, dict):
        return None
    token = str(raw.get("token") or "").strip()
    if not token:
        return None
    return Credentials(token=token)


def save_token(token: str) -> None:
    tok = token.strip()
    if not tok:
        raise ValueError("token must be non-empty")
    path = _credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"token": tok}, indent=2), encoding="utf-8")
    # The token is a long-lived secret; keep it owner-only on POSIX. Windows
    # inherits the user profile ACL, so chmod is a no-op / skipped there.
    if os.name != "nt":
        try:
            os.chmod(path.parent, 0o700)
            os.chmod(path, 0o600)
        except OSError:
            pass


def clear_credentials() -> None:
    path = _credentials_path()
    if path.exists():
        path.unlink()

