"""Reader personas for ensemble click-probability estimation.

The Upworthy archive is 2013-2015 Facebook-feed traffic, so the personas are
era-appropriate social-media news readers, chosen to spread age, politics,
media savvy, and clickbait susceptibility rather than to be demographically
exhaustive.
"""

from __future__ import annotations

from .data import Group

PERSONAS: dict[str, str] = {
    "grad_student": (
        "A 24-year-old female sociology grad student, politically progressive, "
        "heavy Facebook user who shares emotional stories about social justice "
        "and inequality with her friends."
    ),
    "suburban_dad": (
        "A 38-year-old male suburban father of two, politically moderate, scrolls "
        "Facebook during his commute, clicks practical tips, lists, and feel-good "
        "family stories."
    ),
    "teacher": (
        "A 55-year-old female public-school teacher, empathetic, drawn to "
        "human-interest stories about kids, teachers, and people overcoming odds."
    ),
    "college_ironist": (
        "A 19-year-old male college student who browses Facebook ironically, "
        "mocks clickbait with his roommates, but still clicks when a curiosity "
        "gap actually lands."
    ),
    "small_biz_owner": (
        "A 45-year-old male small-business owner, conservative-leaning, "
        "skeptical of sensational headlines, clicks mostly news that feels "
        "concrete and consequential."
    ),
    "night_nurse": (
        "A 29-year-old female nurse who scrolls Facebook on night-shift breaks, "
        "clicks celebrity stories, heartwarming videos, and anything that "
        "promises a quick emotional lift."
    ),
    "retired_vet": (
        "A 62-year-old retired male veteran, reads news the traditional way, "
        "distrusts hype, clicks stories about service, sacrifice, and history."
    ),
    "media_pro": (
        "A 33-year-old female digital-marketing professional who recognizes "
        "every clickbait trick, resents being manipulated, but knows exactly "
        "which framings work on her anyway."
    ),
}

SYSTEM = (
    "You are role-playing one specific news reader in 2014 browsing their "
    "Facebook feed. Reader profile: {persona} Stay in character and judge "
    "headlines exactly as this person would."
)

USER = """\
This reader sees the following headlines in their feed, one at a time, next to \
the same picture. For each headline, judged on its own, estimate the \
probability that THIS reader clicks it.

{candidates}

Reply with JSON only, mapping each headline's number to a click probability \
between 0 and 1. Use the full range; probabilities do not need to sum to 1.
Example format: {{"1": 0.07, "2": 0.55, "3": 0.21}}"""


def render_persona(group: Group, order: list[int], persona_id: str) -> tuple[str, str]:
    """Render (system, user) for one persona, candidates in presentation order."""
    lines = [f"{position + 1}. {group.headlines[idx]}" for position, idx in enumerate(order)]
    system = SYSTEM.format(persona=PERSONAS[persona_id])
    return system, USER.format(candidates="\n".join(lines))
