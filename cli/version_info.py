from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def cli_version() -> str:
    try:
        return version("stockvisionz-cli")
    except PackageNotFoundError:
        return "0.0.0+dev"
