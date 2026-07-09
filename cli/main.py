from __future__ import annotations

import argparse
import json
import time
import webbrowser
from datetime import date
from typing import Any, NoReturn

from cli.credentials import clear_credentials, credentials_path, load_credentials, save_token
from cli.dates import PRESET_IDS, preset_range
from cli.http_client import NETWORK_ERROR_STATUS, ApiResponse, _base_url, get, post
from cli.prompts import confirm_save, should_auto_save, should_prompt_save
from cli.results import default_web_url, format_backtest_summary, format_job_list
from cli.version_info import cli_version

CLI_EPILOG = """
Examples:
  stockvisionz login
  stockvisionz whoami
  stockvisionz run --symbol AAPL --preset 1y
  stockvisionz run --symbol AAPL --start 2023-01-01 --end 2024-01-01 --save
  stockvisionz ingest AAPL
  stockvisionz jobs list
  stockvisionz jobs show 42
  stockvisionz save 42 --name "My run"
"""

RUN_EPILOG = """
Model families: logistic_regression (Contrarian), lightgbm (Whale Watcher)
Validation: rolling (default) or anchored

Examples:
  stockvisionz run --symbol AAPL --preset 1y
  stockvisionz run --symbol AAPL --start 2023-01-27 --end 2023-10-21 --model-family lightgbm
  stockvisionz run --symbol AAPL --preset 6m --save --name "AAPL 6M"
  stockvisionz run --symbol AAPL --start 2023-01-01 --end 2024-01-01 --json --no-save
"""


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def _require_token() -> str:
    creds = load_credentials()
    if not creds:
        raise SystemExit("Not logged in. Run: stockvisionz login")
    return creds.token


def _error_detail(payload: Any) -> str | None:
    """Pull a human message out of the API's `{"error": ...}` shape."""
    if isinstance(payload, dict):
        for key in ("error", "message", "raw"):
            val = payload.get(key)
            if val:
                return str(val)
    elif isinstance(payload, str) and payload:
        return payload
    return None


def _fail(context: str, res: ApiResponse) -> NoReturn:
    """Surface an API failure as a single friendly line, never a raw dict/traceback."""
    detail = _error_detail(res.json)
    if res.status == NETWORK_ERROR_STATUS:
        raise SystemExit(detail or f"{context}: could not reach the StockVisionz API.")
    if res.status == 401:
        raise SystemExit("Not logged in or your token expired. Run: stockvisionz login")
    if detail:
        raise SystemExit(f"{context}: {detail} (HTTP {res.status})")
    raise SystemExit(f"{context}: HTTP {res.status}")


def resolve_run_window(args: argparse.Namespace) -> tuple[date, date]:
    if args.preset:
        return preset_range(args.preset)
    if args.start is None or args.end is None:
        raise SystemExit("Provide --start and --end, or use --preset (6m, 1y, ytd, max).")
    if args.start > args.end:
        raise SystemExit("--start must be on or before --end.")
    return args.start, args.end


def cmd_auth_set_token(args: argparse.Namespace) -> int:
    save_token(args.token)
    print("Saved token.")
    return 0


def cmd_auth_logout(args: argparse.Namespace) -> int:
    _ = args
    creds = load_credentials()
    if creds:
        # Best-effort server-side revocation so a copied token stops working;
        # never block local logout on a network/API failure.
        try:
            res = post("/v1/auth/revoke", token=creds.token)
            if res.status == 200 and isinstance(res.json, dict) and res.json.get("revoked"):
                print("Revoked server token.")
        except Exception:  # noqa: BLE001
            pass
    clear_credentials()
    print("Logged out.")
    return 0


def cmd_whoami(args: argparse.Namespace) -> int:
    token = _require_token()
    res = get("/v1/me", token=token)
    if res.status != 200 or not isinstance(res.json, dict):
        _fail("Failed to fetch account", res)
    if getattr(args, "json", False):
        print(json.dumps(res.json, indent=2))
        return 0
    email = res.json.get("email") or "—"
    username = res.json.get("username") or "—"
    print(f"User ID:  {res.json.get('userId')}")
    print(f"Username: {username}")
    print(f"Email:    {email}")
    return 0


def cmd_auth_status(args: argparse.Namespace) -> int:
    _ = args
    creds = load_credentials()
    print(f"Credentials: {credentials_path()}")
    if creds:
        print("Status: logged in")
        return 0
    print("Status: not logged in")
    print("Run: stockvisionz login")
    return 1


def cmd_version(args: argparse.Namespace) -> int:
    _ = args
    print(f"stockvisionz-cli {cli_version()}")
    print(f"API: {_base_url()}")
    print(f"Web: {default_web_url()}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    _ = args
    creds = load_credentials()
    print(f"config_dir: {credentials_path().parent}")
    print(f"credentials: {credentials_path()}")
    print(f"logged_in: {'yes' if creds else 'no'}")
    print(f"api_url: {_base_url()}")
    print(f"web_url: {default_web_url()}")
    return 0


def cmd_login(args: argparse.Namespace) -> int:
    _ = args
    started = post("/v1/auth/device/start", token="", json_body={})
    if started.status != 200 or not isinstance(started.json, dict):
        _fail("Failed to start device login", started)
    device_code = str(started.json.get("deviceCode") or "")
    user_code = str(started.json.get("userCode") or "")
    verification_url = str(started.json.get("verificationUrl") or "")
    interval = int(started.json.get("interval") or 2)
    expires_in = int(started.json.get("expiresIn") or 600)
    if not device_code or not user_code or not verification_url:
        raise SystemExit("Device login returned an invalid payload")

    print(f"Approve in your browser (code {user_code}):")
    print(f"  {verification_url}")
    # Try to open the browser automatically; harmless if headless (URL is printed above).
    try:
        webbrowser.open(verification_url)
    except Exception:  # noqa: BLE001
        pass
    print("Waiting for approval... (Ctrl-C to cancel)")

    deadline = time.time() + max(expires_in, interval)
    try:
        while True:
            res = get("/v1/auth/device/token", token="", query={"deviceCode": device_code})
            if res.status == 202:
                if time.time() >= deadline:
                    raise SystemExit("Login timed out. Run stockvisionz login again.")
                time.sleep(interval)
                continue
            if res.status != 200 or not isinstance(res.json, dict):
                _fail("Login failed", res)
            token = str(res.json.get("accessToken") or "")
            if not token:
                raise SystemExit("Login failed: missing accessToken")
            save_token(token)
            print("Logged in.")
            return 0
    except KeyboardInterrupt:
        raise SystemExit("\nLogin cancelled.")


def _poll_job(*, token: str, kind: str, job_id: int, timeout_sec: int = 900) -> dict[str, Any]:
    started = time.time()
    while True:
        if kind == "ingest":
            res = get(f"/v1/market-ingest/{job_id}", token=token)
        else:
            res = get(f"/v1/lab/backtest/{job_id}", token=token)
        if res.status != 200 or not isinstance(res.json, dict):
            _fail(f"Failed to fetch job {job_id}", res)
        status = str(res.json.get("status") or "")
        if status in ("complete", "already_available", "failed"):
            return res.json
        if time.time() - started > timeout_sec:
            raise RuntimeError(f"Timed out waiting for job {job_id} (status={status})")
        time.sleep(2)


def _enqueue_ingest(*, token: str, symbol: str) -> dict[str, Any]:
    ingest = post("/v1/market-ingest", token=token, json_body={"symbol": symbol})
    if ingest.status not in (200, 201) or not isinstance(ingest.json, dict):
        _fail("Failed to enqueue ingest", ingest)
    ingest_job_id = int(ingest.json["jobId"])
    print(f"Ingest job: {ingest_job_id} ({ingest.json.get('status')})")
    ingest_final = _poll_job(token=token, kind="ingest", job_id=ingest_job_id)
    if ingest_final.get("status") == "failed":
        raise SystemExit(f"Ingest failed: {ingest_final.get('error')}")
    print(f"Ingest ready: {ingest_final.get('status')} {ingest_final.get('lastBarDate')}")
    return ingest_final


def cmd_ingest(args: argparse.Namespace) -> int:
    token = _require_token()
    symbol = args.symbol.strip().upper()
    if args.json:
        ingest = post("/v1/market-ingest", token=token, json_body={"symbol": symbol})
        print(json.dumps({"status": ingest.status, "body": ingest.json}, indent=2))
        return 0 if ingest.status in (200, 201) else 1
    _enqueue_ingest(token=token, symbol=symbol)
    return 0


def _format_job_status(payload: dict[str, Any]) -> str:
    lines = [
        f"Job {payload.get('jobId', '?')}: {payload.get('status', 'unknown')}",
    ]
    if payload.get("statusMessage"):
        lines.append(f"Message: {payload['statusMessage']}")
    if payload.get("error"):
        lines.append(f"Error: {payload['error']}")
    result = payload.get("result")
    if isinstance(result, dict) and payload.get("status") == "complete":
        lines.append("")
        lines.append(format_backtest_summary(result, job_id=int(payload.get("jobId") or 0)))
    return "\n".join(lines)


def cmd_jobs_show(args: argparse.Namespace) -> int:
    token = _require_token()
    res = get(f"/v1/lab/backtest/{args.job_id}", token=token)
    if res.status != 200 or not isinstance(res.json, dict):
        _fail("Failed to fetch job", res)
    if args.json:
        print(json.dumps(res.json, indent=2))
    else:
        print(_format_job_status(res.json))
    status = str(res.json.get("status") or "")
    return 1 if status == "failed" else 0


def cmd_jobs_list(args: argparse.Namespace) -> int:
    token = _require_token()
    limit = max(1, min(int(getattr(args, "limit", 20) or 20), 100))
    res = get("/v1/lab/backtest", token=token, query={"limit": str(limit)})
    if res.status != 200 or not isinstance(res.json, dict):
        _fail("Failed to list jobs", res)
    jobs = res.json.get("jobs")
    jobs = jobs if isinstance(jobs, list) else []
    if getattr(args, "json", False):
        print(json.dumps({"jobs": jobs}, indent=2))
    else:
        print(format_job_list(jobs))
    return 0


def cmd_save(args: argparse.Namespace) -> int:
    token = _require_token()
    _save_run_to_account(
        token=token, job_id=args.job_id, name=args.name, json_mode=getattr(args, "json", False)
    )
    return 0


def _bars_to_df(symbol: str, bars_payload: dict[str, Any]):
    import pandas as pd

    bars = bars_payload.get("bars")
    if not isinstance(bars, list):
        raise RuntimeError("API returned invalid bars payload")
    rows = []
    for b in bars:
        if not isinstance(b, dict):
            continue
        rows.append(
            {
                "symbol": symbol,
                "bar_date": date.fromisoformat(str(b["barDate"])),
                "open": float(b["open"]),
                "high": float(b["high"]),
                "low": float(b["low"]),
                "close": float(b["close"]),
                "volume": int(b["volume"]) if b.get("volume") is not None else None,
                "sma_20": float(b["sma20"]) if b.get("sma20") is not None else None,
                "ema": float(b["ema"]) if b.get("ema") is not None else None,
                "adx": float(b["adx"]) if b.get("adx") is not None else None,
                "rsi": float(b["rsi"]) if b.get("rsi") is not None else None,
                "macd_line": float(b["macdLine"]) if b.get("macdLine") is not None else None,
                "macd_signal": float(b["macdSignal"]) if b.get("macdSignal") is not None else None,
            }
        )
    return pd.DataFrame(rows)


def _save_run_to_account(
    *, token: str, job_id: int, name: str | None, json_mode: bool = False
) -> None:
    body: dict[str, Any] = {}
    if name:
        body["name"] = name
    save_res = post(f"/v1/lab/backtest/{job_id}/save", token=token, json_body=body)
    if save_res.status != 200 or not isinstance(save_res.json, dict):
        _fail("Failed to save run", save_res)
    if json_mode:
        print(json.dumps(save_res.json, indent=2))
        return
    run_id = save_res.json.get("backtestRunId")
    message = str(save_res.json.get("message") or "Saved to your account")
    web_url = default_web_url()
    print(f"{message} (run #{run_id}). View in Lab: {web_url}/dashboard/lab")


def _handle_post_run_save(args: argparse.Namespace, *, token: str, job_id: int) -> None:
    if args.no_save or args.json:
        return
    if should_auto_save(auto_save=args.save, no_save=args.no_save, json_mode=args.json):
        _save_run_to_account(token=token, job_id=job_id, name=args.name)
        return
    if should_prompt_save(auto_save=args.save, no_save=args.no_save, json_mode=args.json):
        if confirm_save():
            _save_run_to_account(token=token, job_id=job_id, name=args.name)
        return
    print("Skipped save (non-interactive). Pass --save to persist without a prompt.")


def cmd_run(args: argparse.Namespace) -> int:
    try:
        from ml.lab_backtest import run_lab_backtest
    except ImportError:
        raise SystemExit(
            "Local backtest engine failed to import. "
            "Try reinstalling: pipx install --force stockvisionz-cli"
        )

    token = _require_token()
    symbol = args.symbol.strip().upper()
    start, end = resolve_run_window(args)

    _enqueue_ingest(token=token, symbol=symbol)

    print(f"Running local backtest: {start.isoformat()} → {end.isoformat()}")
    backtest = post(
        "/v1/lab/backtest",
        token=token,
        json_body={
            "symbol": symbol,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "modelFamily": args.model_family,
            "validationMode": args.validation_mode,
            "runner": "local",
            "strategyParams": {},
        },
    )
    if backtest.status not in (202, 200) or not isinstance(backtest.json, dict):
        _fail("Failed to create backtest job", backtest)
    job_id = int(backtest.json["jobId"])
    print(f"Backtest job: {job_id} (local)")

    bars_res = get(
        "/v1/lab/backtest/data",
        token=token,
        query={"symbol": symbol, "startDate": start.isoformat(), "endDate": end.isoformat()},
    )
    if bars_res.status != 200 or not isinstance(bars_res.json, dict):
        _fail("Failed to fetch bars", bars_res)
    df = _bars_to_df(symbol, bars_res.json)

    try:
        payload = run_lab_backtest(
            symbol,
            start,
            end,
            model_family=args.model_family,
            validation_mode=args.validation_mode,
            bars=df,
        )
        complete_body: dict[str, Any] = {
            "status": "complete",
            "statusMessage": "Backtest complete (local)",
            "result": payload,
        }
    except Exception as exc:  # noqa: BLE001
        complete_body = {
            "status": "failed",
            "statusMessage": f"Backtest failed (local): {exc}",
            "error": str(exc),
        }

    done = post(f"/v1/lab/backtest/{job_id}/complete", token=token, json_body=complete_body)
    if done.status != 200:
        _fail("Failed to complete job", done)

    if complete_body["status"] == "failed":
        print(f"Backtest failed (job {job_id}).")
        msg = complete_body.get("statusMessage") or complete_body.get("error")
        if msg:
            print(msg)
        return 1

    payload = complete_body["result"]
    assert isinstance(payload, dict)

    if args.json:
        print(json.dumps({"jobId": job_id, "result": payload}, indent=2))
    else:
        print(format_backtest_summary(payload, job_id=job_id))
        _handle_post_run_save(args, token=token, job_id=job_id)

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stockvisionz",
        description="StockVisionz CLI — local Lab backtests synced to your account.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLI_EPILOG,
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {cli_version()}",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    auth = sub.add_parser("auth", help="Authentication utilities")
    auth_sub = auth.add_subparsers(dest="auth_cmd", required=True)
    st = auth_sub.add_parser("set-token", help="Store a Clerk Bearer token (advanced)")
    st.add_argument("token")
    st.set_defaults(func=cmd_auth_set_token)
    lo = auth_sub.add_parser("logout", help="Remove stored credentials")
    lo.set_defaults(func=cmd_auth_logout)
    ast = auth_sub.add_parser("status", help="Show whether you are logged in")
    ast.set_defaults(func=cmd_auth_status)

    login = sub.add_parser("login", help="Connect this CLI to your account")
    login.set_defaults(func=cmd_login)

    logout = sub.add_parser("logout", help="Remove stored credentials")
    logout.set_defaults(func=cmd_auth_logout)

    version_cmd = sub.add_parser("version", help="Show CLI version and API endpoint")
    version_cmd.set_defaults(func=cmd_version)

    config = sub.add_parser("config", help="Show config paths and environment")
    config.set_defaults(func=cmd_config)

    whoami = sub.add_parser("whoami", help="Show the account this CLI is linked to")
    whoami.add_argument("--json", action="store_true", help="Print raw account JSON")
    whoami.set_defaults(func=cmd_whoami, json=False)

    ingest = sub.add_parser("ingest", help="Fetch or update market bars for a symbol")
    ingest.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    ingest.add_argument("--json", action="store_true", help="Print raw API response only")
    ingest.set_defaults(func=cmd_ingest, json=False)

    jobs = sub.add_parser("jobs", help="Inspect backtest jobs")
    jobs_sub = jobs.add_subparsers(dest="jobs_cmd", required=True)
    jobs_show = jobs_sub.add_parser("show", help="Show a backtest job by ID")
    jobs_show.add_argument("job_id", type=int)
    jobs_show.add_argument("--json", action="store_true", help="Print raw job JSON")
    jobs_show.set_defaults(func=cmd_jobs_show, json=False)
    jobs_list = jobs_sub.add_parser("list", help="List your recent backtest jobs")
    jobs_list.add_argument("--limit", type=int, default=20, help="Max jobs to show (1-100)")
    jobs_list.add_argument("--json", action="store_true", help="Print raw jobs JSON")
    jobs_list.set_defaults(func=cmd_jobs_list, json=False, limit=20)

    save = sub.add_parser("save", help="Save a completed backtest job to your account")
    save.add_argument("job_id", type=int, help="Backtest job ID from a prior run")
    save.add_argument("--name", default=None, help="Optional display name")
    save.add_argument("--json", action="store_true", help="Print raw save response JSON")
    save.set_defaults(func=cmd_save, json=False)

    run = sub.add_parser(
        "run",
        help="Run a local Lab backtest and sync results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=RUN_EPILOG,
    )
    run.add_argument("--symbol", required=True, help="Ticker symbol, e.g. AAPL")
    run.add_argument("--start", type=_parse_date, help="Start date (YYYY-MM-DD)")
    run.add_argument("--end", type=_parse_date, help="End date (YYYY-MM-DD)")
    run.add_argument(
        "--preset",
        choices=list(PRESET_IDS),
        help="Date preset instead of --start/--end (6m, 1y, ytd, max)",
    )
    run.add_argument(
        "--model-family",
        default="logistic_regression",
        choices=["logistic_regression", "lightgbm"],
        help="Model family (default: logistic_regression)",
    )
    run.add_argument(
        "--validation-mode",
        default="rolling",
        choices=["rolling", "anchored"],
        help="Walk-forward validation mode (default: rolling)",
    )
    save_group = run.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save",
        action="store_true",
        help="Save the run to your account without prompting",
    )
    save_group.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the run to your account",
    )
    run.add_argument(
        "--json",
        action="store_true",
        help="Print JSON result to stdout (no save prompt)",
    )
    run.add_argument(
        "--name",
        default=None,
        help="Optional display name when saving to your account",
    )
    run.set_defaults(func=cmd_run, save=False, no_save=False, json=False, preset=None)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    fn = getattr(args, "func", None)
    if not fn:
        parser.print_help()
        return 2
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
