"""Upworthy Research Archive -> within-group headline-ranking benchmark.

The archive (Matias, Munger et al.) is a public record of A/B tests Upworthy ran
on its own headlines between 2013 and 2015. Each row is one *package*: a
headline shown with an image, plus the impressions and clicks it received. Rows
that share a test were shown to randomly assigned visitors, so their click-through
rates are directly comparable.

That comparability is the whole point. It gives a ranking task with a real,
randomised ground truth: given K headlines that competed against each other,
put them in click-through order.

Data is fetched from OSF at first use; nothing is redistributed here.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# The per-split OSF nodes carry the 12.14.2023 release, which is the only one
# with the `problem` randomisation flag. The umbrella node's 03.12.2020 files
# predate that correction; do not substitute them.
SOURCES = {
    "exploratory": (
        "https://osf.io/download/67jvw/",
        "upworthy-archive-exploratory-packages-12.14.2023.csv",
    ),
    "holdout": (
        "https://osf.io/download/e3k5j/",
        "upworthy-archive-holdout-packages-12.14.2023.csv",
    ),
    "confirmatory": (
        "https://osf.io/download/h36sa/",
        "upworthy-archive-confirmatory-packages-12.14.2023.csv",
    ),
}

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# --- filtering constants -----------------------------------------------------
# A package needs enough traffic for its CTR to mean anything. 500 is the floor
# used throughout; the group count is insensitive to moving it to 1000.
MIN_IMPRESSIONS = 500
# A group needs at least 3 distinct headlines for a rank correlation to carry
# information. K=2 degenerates to a coin flip.
MIN_CANDIDATES = 3
# Upworthy's randomisation is documented as unreliable in this window.
RISK_WINDOW = ("2013-06-25", "2014-01-10")


@dataclass(frozen=True)
class Group:
    """One evaluation group: K headlines that competed under identical conditions."""

    group_id: str
    headlines: tuple[str, ...]
    ctr: tuple[float, ...]
    impressions: tuple[int, ...]

    @property
    def k(self) -> int:
        return len(self.headlines)


def normalize(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().lower()


def download(split: str) -> Path:
    url, name = SOURCES[split]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / name
    if not target.exists():
        print(f"downloading {split} from {url} ...")
        with urllib.request.urlopen(url) as response, target.open("wb") as out:  # noqa: S310
            out.write(response.read())
    return target


def load_groups(split: str = "holdout") -> dict[str, Group]:
    """Build ranking groups from one archive split, or "all" for every split.

    ``holdout`` is by far the largest of the three (105k packages, ~8.4k usable
    groups); ``exploratory`` and ``confirmatory`` hold roughly 1.7k and 1.8k.
    The names are the archive's own pre-registration split, unrelated to the
    train/eval split this study makes.

    A group is ``(clickability_test_id, eyecatcher_id, normalized excerpt)``.

    Grouping at that sub-test level rather than at the test level matters: a
    single Upworthy test often varied the image as well as the headline, and
    comparing headlines across different images confounds the thing being
    measured. Splitting on the image and excerpt keeps every within-group
    comparison a clean headline-vs-headline contrast, and recovers several
    hundred usable groups that test-level grouping discards wholesale.
    """
    if split == "all":
        frames = [pd.read_csv(download(name), low_memory=False) for name in SOURCES]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(download(split), low_memory=False)

    required = {"clickability_test_id", "headline", "impressions", "clicks"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"archive is missing expected columns: {sorted(missing)}")

    n0 = len(df)
    if "problem" in df.columns:
        # Boolean in the 12.14.2023 release, but it has shipped as a string flag
        # and as a blank-or-reason column in other releases. Treat anything that
        # is not an explicit falsehood as a problem.
        flag = df["problem"].astype(str).str.strip().str.lower()
        df = df[flag.isin({"false", "0", "nan", "", "none"})]
    if "created_at" in df.columns:
        created = pd.to_datetime(df["created_at"], errors="coerce")
        lo, hi = pd.Timestamp(RISK_WINDOW[0]), pd.Timestamp(RISK_WINDOW[1])
        df = df[~created.between(lo, hi)]

    df["headline_norm"] = df["headline"].map(normalize)
    df = df[df["headline_norm"] != ""]
    df = df[df["impressions"] >= MIN_IMPRESSIONS]
    df = df[df["clicks"] >= 0]

    image = df.get("eyecatcher_id", pd.Series("", index=df.index)).fillna("").astype(str)
    df = df[image.str.strip() != ""]
    excerpt = df.get("excerpt", pd.Series("", index=df.index)).map(normalize)

    df["group_id"] = (
        df["clickability_test_id"].astype(str)
        + "|"
        + image.str.strip()
        + "|"
        + excerpt.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:8])
    )

    # Within a group the same headline can appear twice (Upworthy re-ran
    # packages). Pool them rather than keeping one arbitrarily: summed
    # impressions give the least noisy estimate of that headline's CTR.
    pooled = (
        df.groupby(["group_id", "headline_norm"])
        .agg(
            headline=("headline", "first"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
        )
        .reset_index()
    )
    pooled["ctr"] = pooled["clicks"] / pooled["impressions"]

    sizes = pooled.groupby("group_id")["headline_norm"].nunique()
    pooled = pooled[pooled["group_id"].map(sizes) >= MIN_CANDIDATES]

    groups: dict[str, Group] = {}
    for gid, block in pooled.groupby("group_id"):
        block = block.sort_values("headline_norm")
        ctr = tuple(float(v) for v in block["ctr"])
        # A group where every headline scored identically carries no ranking
        # signal and would contribute an undefined correlation.
        if np.allclose(ctr, ctr[0]):
            continue
        groups[str(gid)] = Group(
            group_id=str(gid),
            headlines=tuple(str(h) for h in block["headline"]),
            ctr=ctr,
            impressions=tuple(int(v) for v in block["impressions"]),
        )

    print(f"{split}: {n0} rows -> {len(groups)} groups (K>={MIN_CANDIDATES})")
    return groups


def presentation_order(namespace: str, group_id: str, k: int) -> list[int]:
    """A fixed but content-derived permutation of a group's candidates.

    Position bias is real, so the order a model sees must not be the order the
    ground truth is stored in. Deriving it from a hash keeps it stable across
    runs without a global seed, and the namespace keeps training-set orders
    disjoint from evaluation orders.
    """
    seed = int.from_bytes(
        hashlib.sha256(f"{namespace}|{group_id}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    return list(rng.permutation(k))


def confident(group: Group) -> bool:
    """Is this group's ordering statistically solid enough to train on?

    Requires real traffic, a CTR spread that is large relative to the mean, and
    a top-vs-bottom gap that clears binomial noise at roughly z=2.

    This filter exists because of the v1 result: training on every group taught
    the model to reproduce coin-flip orderings as though they were signal. See
    RESULTS.md.
    """
    ctr = np.asarray(group.ctr, dtype=float)
    imp = np.asarray(group.impressions, dtype=float)
    if imp.min() < 1000:
        return False
    spread = (ctr.max() - ctr.min()) / max(ctr.mean(), 1e-12)
    p_hi, p_lo = float(ctr.max()), float(ctr.min())
    n_hi, n_lo = float(imp[ctr.argmax()]), float(imp[ctr.argmin()])
    se = (p_hi * (1 - p_hi) / n_hi + p_lo * (1 - p_lo) / n_lo) ** 0.5
    z = (p_hi - p_lo) / max(se, 1e-12)
    return bool(spread >= 0.25 and z >= 2.0)
