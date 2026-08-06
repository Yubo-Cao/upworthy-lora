"""Load the Upworthy Research Archive and build ranking groups."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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

MIN_IMPRESSIONS = 500
MIN_CANDIDATES = 3
RISK_WINDOW = ("2013-06-25", "2014-01-10")


@dataclass(frozen=True)
class Group:
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
    """Build ranking groups from one archive split, or "all" for every split."""
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
    """Hash-based permutation so training and eval see different orderings."""
    seed = int.from_bytes(
        hashlib.sha256(f"{namespace}|{group_id}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    return list(rng.permutation(k))


def confident(group: Group) -> bool:
    """v2 filter: keep only groups whose ordering clears binomial noise."""
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
