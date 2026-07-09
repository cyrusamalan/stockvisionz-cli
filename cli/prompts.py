from __future__ import annotations

import sys


def confirm_save(prompt: str = "Save this run to your StockVisionz account? [y/N] ") -> bool:
    """Return True when the user explicitly confirms with y/yes."""
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def should_prompt_save(*, auto_save: bool, no_save: bool, json_mode: bool) -> bool:
    """Decide whether to show the interactive save prompt."""
    if auto_save or no_save or json_mode:
        return False
    return sys.stdin.isatty()


def should_auto_save(*, auto_save: bool, no_save: bool, json_mode: bool) -> bool:
    """Decide whether to save without prompting."""
    if no_save or json_mode:
        return False
    if auto_save:
        return True
    return False
