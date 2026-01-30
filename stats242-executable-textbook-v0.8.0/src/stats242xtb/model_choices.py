from __future__ import annotations
from typing import Any, Dict

def model_card(title: str, choices: Dict[str, Any], notes: list[str] | None = None) -> str:
    lines = [f"**{title} — Model choices**", ""]
    for k, v in choices.items():
        lines.append(f"- **{k}**: {v}")
    if notes:
        lines += ["", "**Notes**"]
        for n in notes:
            lines.append(f"- {n}")
    return "\n".join(lines) + "\n"
