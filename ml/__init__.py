"""
StockVisionz ML models.

Trainable "raw" models the user runs over train/test periods they choose, plus a
reusable backtest/metrics layer and a market_bar reader. The first implemented model
is Mean-Reversion Drop, *The Contrarian* (logistic regression); see
`ml_models_deep_dive (claude).md` §1 for the build spec.
"""
