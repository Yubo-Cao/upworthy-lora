"""Prompt template for headline click-probability estimation."""

from __future__ import annotations

from .data import Group

SYSTEM = "You estimate how likely news headlines are to be clicked by ordinary readers."

USER = """\
A random visitor to a general-interest news site sees the following headlines, \
one at a time, next to the same picture. For each headline, judged on its own, \
estimate the probability that the visitor clicks it.

{candidates}

Reply with JSON only, mapping each headline's number to a click probability \
between 0 and 1. Use the full range; probabilities do not need to sum to 1.
Example format: {{"1": 0.07, "2": 0.55, "3": 0.21}}"""


def render(group: Group, order: list[int]) -> tuple[str, str]:
    """Render (system, user) with candidates in the given presentation order."""
    lines = [f"{position + 1}. {group.headlines[idx]}" for position, idx in enumerate(order)]
    return SYSTEM, USER.format(candidates="\n".join(lines))
