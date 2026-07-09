from __future__ import annotations

import calendar
from datetime import date

PRESET_IDS = ("6m", "1y", "ytd", "max")


def subtract_months(value: date, months: int) -> date:
    month = value.month - months
    year = value.year
    while month <= 0:
        month += 12
        year -= 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(value.day, last_day))


def preset_range(preset: str, today: date | None = None) -> tuple[date, date]:
    if preset not in PRESET_IDS:
        raise ValueError(f"unknown preset: {preset}")
    end = today or date.today()
    if preset == "ytd":
        return date(end.year, 1, 1), end
    if preset == "max":
        return date(max(end.year - 10, 2000), 1, 1), end
    months = 6 if preset == "6m" else 12
    return subtract_months(end, months), end
