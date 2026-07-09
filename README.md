# StockVisionz CLI

Run [StockVisionz](https://stockvisionz.com) Lab backtests on your own machine and sync the results to
your account. Backtests execute locally (the model runs on your CPU); everything else talks to the
StockVisionz API.

## Install

```bash
pipx install stockvisionz-cli
```

The machine-learning dependencies (`pandas`, `numpy`, `scikit-learn`, `lightgbm`) are bundled, so
`stockvisionz run` works right after install.

## Log in

```bash
stockvisionz login
```

This opens a browser to approve the device against your StockVisionz account (a device-code flow — no
password is entered in the terminal). Once approved, the CLI stores a token under your OS config dir
(`~/.config/stockvisionz/credentials.json`, or `%APPDATA%\stockvisionz\` on Windows; permissions are
tightened to owner-only on POSIX).

```bash
stockvisionz whoami        # confirm the linked account
```

## Run a backtest

```bash
# Preset window
stockvisionz run --symbol AAPL --preset 1y

# Explicit window, save the run to your account
stockvisionz run --symbol AAPL --start 2023-01-27 --end 2023-10-21 --save

# JSON output (no save prompt)
stockvisionz run --symbol AAPL --preset 6m --json
```

Model families: `logistic_regression` (Contrarian) and `lightgbm` (Whale Watcher), via
`--model-family`. Walk-forward validation is `--validation-mode rolling` (default) or `anchored`.

## Manage data & runs

```bash
stockvisionz ingest AAPL              # fetch/refresh market bars for a symbol
stockvisionz jobs list                # your recent backtest jobs
stockvisionz jobs show <jobId>        # details for one job
stockvisionz save <jobId> --name "My run"
stockvisionz logout                   # revoke the token server-side and clear it locally
```

Other: `stockvisionz version`, `stockvisionz config`, `stockvisionz auth status`.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `STOCKVISIONZ_API_URL` | Override the API base URL | the hosted API |
| `STOCKVISIONZ_API_TIMEOUT` | HTTP timeout (seconds) | `60` |
| `STOCKVISIONZ_CONFIG_DIR` | Where credentials are stored | OS config dir |
| `STOCKVISIONZ_WEB_URL` | Web app base URL for printed links | `https://stockvisionz.com` |

## How it works

1. `login` runs an OAuth-style device flow: the CLI asks the API for a code, you approve it in the
   browser, and the CLI exchanges it for a long-lived API token (stored hashed server-side).
2. `run` enqueues a market-data ingest for the symbol, waits until bars are ready, pulls the daily
   OHLCV bars from the API, runs the walk-forward backtest **locally**, then posts the completed result
   back to your account (optionally saving it).

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

## License

MIT — see [LICENSE](LICENSE).
