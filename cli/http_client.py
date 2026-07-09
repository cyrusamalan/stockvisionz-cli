from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from cli.api_config import DEFAULT_API_URL
from cli.version_info import cli_version


JSON = dict[str, Any] | list[Any] | str | int | float | bool | None

# ApiResponse.status == 0 is the CLI-internal sentinel for "the request never
# reached the server" (DNS failure, connection refused, TLS error, timeout).
NETWORK_ERROR_STATUS = 0

DEFAULT_TIMEOUT = 60.0


def _base_url() -> str:
    base = (os.environ.get("STOCKVISIONZ_API_URL") or "").strip()
    if base:
        return base.rstrip("/")
    return DEFAULT_API_URL


def _timeout() -> float:
    raw = (os.environ.get("STOCKVISIONZ_API_TIMEOUT") or "").strip()
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT
    return value if value > 0 else DEFAULT_TIMEOUT


@dataclass(frozen=True)
class ApiResponse:
    status: int
    json: JSON


def _request(
    method: str,
    path: str,
    *,
    token: str,
    json_body: JSON | None = None,
    query: dict[str, str] | None = None,
) -> ApiResponse:
    base = _base_url()
    url = f"{base}{path}"
    if query:
        from urllib.parse import urlencode

        url = f"{url}?{urlencode(query)}"

    data: bytes | None
    headers = {
        "user-agent": f"stockvisionz-cli/{cli_version()}",
        "accept": "application/json",
    }
    if token:
        headers["authorization"] = f"Bearer {token}"
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["content-type"] = "application/json; charset=utf-8"
    else:
        data = None

    req = urllib.request.Request(url=url, data=data, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_timeout()) as res:
            raw = res.read() if res.readable() else b""
            body = raw.decode("utf-8") if raw else ""
            try:
                payload = json.loads(body) if body else None
            except Exception:  # noqa: BLE001
                payload = {"raw": body}
            return ApiResponse(status=res.status, json=payload)
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8") if err.fp else ""
        try:
            payload = json.loads(body) if body else None
        except Exception:  # noqa: BLE001
            payload = {"raw": body}
        return ApiResponse(status=int(err.code), json=payload)
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as err:
        # Never reached the server: return a sentinel instead of a raw traceback.
        reason = getattr(err, "reason", None) or err
        return ApiResponse(
            status=NETWORK_ERROR_STATUS,
            json={"error": f"Could not reach the StockVisionz API at {base}: {reason}"},
        )


def get(path: str, *, token: str, query: dict[str, str] | None = None) -> ApiResponse:
    return _request("GET", path, token=token, query=query)


def post(path: str, *, token: str, json_body: JSON | None = None) -> ApiResponse:
    return _request("POST", path, token=token, json_body=json_body)

