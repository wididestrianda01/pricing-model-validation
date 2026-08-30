"""Argument validation shared by the benchmark functions."""

from __future__ import annotations


def check_option_type(option_type: str, allowed: tuple[str, ...]) -> str:
    if option_type not in allowed:
        raise ValueError(f"option_type must be one of {allowed}, got {option_type!r}")
    return option_type

def option_sign(option_type: str) -> float:
    """+1 for a call, -1 for a put; call after `check_option_type` has validated."""
    return 1.0 if option_type == "call" else -1.0



def check_positive(**values: float) -> None:
    """Raise ValueError if any named argument is not strictly positive."""
    for name, value in values.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}")
