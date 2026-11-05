"""
Selection of anomalous rows from an SQL result.

Functionality:
  - Takes the full result set (list[dict], such as `execution_result_full_dict`).
  - Uses ONLY numeric columns (measures) as features, applying robust scaling
    (median/IQR; log1p for highly skewed counts). Categorical columns
    are NOT included as features; they remain as context in the returned rows
    and—if specified in `group_by`—define what is "anomalous relative to the group"
    (e.g., `group_by=["region"]`: 20 events might be normal in Bogotá but
    anomalous in a department with an average of 3).
  - Excludes temporal columns, IDs, constants, and non-measure columns (centroid
    coordinates, sorting/sectioning keys like `sort_*`/`section_*`, `rank_*`);
    these describe location or identity rather than magnitude (time is handled
    via time windows, not clustering).
  - Assigns an anomaly score to each row using IsolationForest
  - Returns a MIXED selection (configurable with environment variables):
      * the TOP_K most anomalous rows (default 50), label `top_anomalous`;
      * the remaining rows, ranked by score, are split into N_TIERS equal-size tiers
        (default 5): tier_1 = just below the top ... tier_N = most typical. From each tier
        the first PER_TIER rows by score are shown (default 10), label `tier_k`.
    Each row carries `_anomaly_score` and `_selection`. The two groups are interleaved, so
    that token truncation of the tail cannot remove one group or one tier entirely.
    A diagnostic dictionary is returned too.
  - Small tables (<= `full_threshold` rows) are passed through unchanged.
"""

import logging
import re
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
# temporal column names: comparison by TOKEN (split on "_"), not by substring:
# "week_start" and "sort_week" are temporal; "weekly_fatalities" and "monthly_events" are measures
_TIME_TOKENS = frozenset({"week", "date", "time", "timestamp", "year", "month", "day", "quarter", "period"})

# columns describing "where/which/in what order," not "how much": never anomaly measurements
_NON_MEASURE_PREFIXES = ("centroid_", "sort_", "section_", "rank_")
_NON_MEASURE_SUFFIXES = ("_rank", "_sort_key")
_NON_MEASURE_EXACT = frozenset({"rank", "rn", "row_num", "row_number", "sort_key"})
_COORD_RE = re.compile(r"(^|_)(lat|lon|lng|latitude|longitude)$")
_MIN_GROUP_SIZE = 20
_Z_CLIP = 20.0

# mixed selection (see module docstring)
_TOP_K = 50
_N_TIERS = 5
_PER_TIER = 10


def _is_non_measure(name: str) -> bool:
    l = name.lower()
    return (
        l in _NON_MEASURE_EXACT
        or l.startswith(_NON_MEASURE_PREFIXES)
        or l.endswith(_NON_MEASURE_SUFFIXES)
        or bool(_COORD_RE.search(l))
    )


def _looks_temporal(name: str, series) -> bool:
    if _TIME_TOKENS & set(re.split(r"[^a-z0-9]+", name.lower())):
        return True
    sample = series.dropna().astype(str).head(20)
    return len(sample) > 0 and all(_DATE_RE.match(s) for s in sample)


def _is_id_like(name: str, series, n: int) -> bool:
    import pandas as pd

    lname = name.lower()
    if lname == "id" or lname.endswith("_id") or lname.endswith("_cnty"):
        return True
    if n >= 50 and series.nunique(dropna=True) / n > 0.95:
        # all distinct values: identifier, except for continuous floats (measurements)
        return not pd.api.types.is_float_dtype(series)
    return False


def _robust_z(v: np.ndarray) -> np.ndarray:
    """(v - median) / robust scale; the scale shifts from the IQR to wider intervals if the data are nearly identical."""
    med = np.median(v)
    for lo, hi in ((25, 75), (10, 90), (1, 99)):
        a, b = np.percentile(v, [lo, hi])
        scale = float(b - a)
        if scale > 0:
            break
    if scale <= 0:
        scale = float(v.std()) or 1.0
    return np.clip((v - med) / scale, -_Z_CLIP, _Z_CLIP)


def build_feature_matrix(
    df, *, exclude: Sequence[str] = (), group_by: Sequence[str] = ()
):
    """Return (X, used_numeric_columns, {discarded_column: reason})."""
    import pandas as pd

    n = len(df)
    dropped: Dict[str, str] = {}
    used: List[str] = []
    cols: List[np.ndarray] = []

    group_cols = [g for g in group_by if g in df.columns]
    groups = df.groupby(group_cols, dropna=False).ngroup().to_numpy() if group_cols else None

    for col in df.columns:
        s = df[col]
        if col in exclude:
            dropped[col] = "excluded"
            continue
        if _is_non_measure(col):
            dropped[col] = "non-measure (coordinates / sort key / rank)"
            continue
        if s.nunique(dropna=True) <= 1:
            dropped[col] = "constant"
            continue
        if _looks_temporal(col, s):
            dropped[col] = "temporal"
            continue
        if _is_id_like(col, s, n):
            dropped[col] = "id-like"
            continue
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            dropped[col] = "grouping-context" if col in group_cols else "categorical (context only)"
            continue

        x = pd.to_numeric(s, errors="coerce").astype(float)
        med = x.median()
        v = x.fillna(0.0 if np.isnan(med) else med).to_numpy()
        # skewness calculated on the "body" of the distribution (p1–p99): the outliers themselves
        # must not determine the transformation, otherwise log1p compresses their signal
        lo, hi = np.percentile(v, [1, 99])
        skew = pd.Series(np.clip(v, lo, hi)).skew()
        if v.min() >= 0 and not np.isnan(skew) and skew > 2:
            v = np.log1p(v)  # asymmetric count

        if groups is None:
            z = _robust_z(v)
        else:
            z = np.full(n, np.nan)
            for gid in np.unique(groups):
                m = groups == gid
                if m.sum() >= _MIN_GROUP_SIZE:
                    z[m] = _robust_z(v[m])
            missing = np.isnan(z)  # groups that are too small: global statistics
            if missing.any():
                z[missing] = _robust_z(v)[missing]

        cols.append(z)
        used.append(col)

    if not cols:
        return None, [], dropped
    return np.column_stack(cols), used, dropped


def score_anomalies(
    X: np.ndarray, random_state: int = 0
):
    """Anomaly score per row (higher = more anomalous)"""
    n = len(X)

    from sklearn.ensemble import IsolationForest

    # high max_samples: with the default (256), a rare anomaly appears in only a few trees,
    # and for a nearly constant feature, the trees without anomalies have no splits on that feature.
    iso = IsolationForest(
        n_estimators=100, max_samples=min(n, 10000), random_state=random_state, n_jobs=-1
    ).fit(X)
    return -iso.score_samples(X)


def select_anomalous_rows(
    rows,
    *,
    full_threshold: int = 50,
    exclude_columns = (),
    group_by = (),
    random_state: int = 0,
):
    """
    Returns (rows_to_show, info). It never raises an exception: in the event of an error or unusable features,
    it returns the original rows.
    """
    rows = rows or []
    n = len(rows)
    info = {
        "n_total": n,
        "n_selected": n,
        "features_used": [],
        "features_dropped": {},
        "group_by": list(group_by),
    }
    if n <= full_threshold:
        return rows, info

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        X, used, dropped = build_feature_matrix(df, exclude=exclude_columns, group_by=group_by)
        info["features_dropped"] = dropped
        if X is None:
            return rows, info

        scores = score_anomalies(X, random_state=random_state)
        order = np.argsort(-scores, kind="stable")  # most anomalous first
        top_k = min(_TOP_K, n)
        top_idx = order[:top_k]
        rest = order[top_k:]  # still sorted by descending score

        # equal-count tiers of the remaining rows (tier_1 = just below the top, tier_N = most typical);
        # from each tier, its first _PER_TIER rows by score.
        tier_picks = []
        tier_info = []
        if len(rest) and _N_TIERS > 0 and _PER_TIER > 0:
            offset = top_k
            for t, chunk in enumerate(np.array_split(rest, _N_TIERS), start=1):
                if len(chunk) == 0:
                    continue
                start, end = offset, offset + len(chunk)
                offset = end
                tier_picks.append(chunk[len(chunk)//2 - 5 : len(chunk)//2 + 5])
                tier_info.append(
                    {
                        "tier": t,
                        "rank_from": start + 1,
                        "rank_to": end,
                        "top_pct_from": round(100.0 * start / n, 1),
                        "top_pct_to": round(100.0 * end / n, 1),
                    }
                )

        def _row(i: int, label: str):
            return {**rows[int(i)], "_anomaly_score": round(float(scores[i]), 3), "_selection": label}

        top_rows = [_row(i, "top_anomalous") for i in top_idx]
        tier_rows = [[_row(i, f"tier_{ti['tier']}") for i in picks] for ti, picks in zip(tier_info, tier_picks)]
        # round-robin across tiers, then 1:1 with the top rows: truncating the tail keeps every group represented
        representative = [r for grp in zip_longest(*tier_rows) for r in grp if r is not None]
        selected = []
        for a, r in zip_longest(top_rows, representative):
            if a is not None:
                selected.append(a)
            if r is not None:
                selected.append(r)

        if len(selected) >= n:  # nothing would be left out: show the rows as they are
            return rows, info
        info.update(
            n_selected=len(selected),
            n_top=len(top_rows),
            n_representative=len(representative),
            per_tier=_PER_TIER,
            tiers=tier_info,
            features_used=used,
        )
        return selected, info
    except Exception as e:  # noqa: BLE001
        logger.warning("selection of anomalous rows failed, using all rows: %s", e)
        info["error"] = str(e)
        return rows, info


def describe_selection(info) -> str:
    """Note to precede the table shown in the template. Empty string if no selection was made."""
    if info.get("n_selected", 0) >= info.get("n_total", 0):
        return ""
    used = ", ".join(info.get("features_used", [])) or "n/a"
    grouping = (
        f", relative to each {'/'.join(info['group_by'])}" if info.get("group_by") else ""
    )
    tiers = info.get("tiers") or []
    if tiers:
        tier_txt = "; ".join(
            f"tier_{t['tier']} = rows ranked {t['rank_from']}-{t['rank_to']} "
            f"(top {t['top_pct_from']}%-{t['top_pct_to']}% most unusual)"
            for t in tiers
        )
        tier_note = (
            f"- `tier_1` ... `tier_{len(tiers)}`: the remaining rows, ranked by unusualness, are split into "
            f"{len(tiers)} equal-size tiers; for each tier the {info.get('per_tier')} highest-score rows are shown. "
            f"tier_1 is just below the most unusual rows and tier_{len(tiers)} is the most typical region of the data "
            f"({tier_txt}).\n"
        )
    else:
        tier_note = ""
    return (
        f"**Note:** the query returned {info['n_total']} rows. Only {info['n_selected']} are shown, marked in the "
        f"`_selection` column (measures used for the score: {used}{grouping}):\n"
        f"- `top_anomalous`: the {info.get('n_top', info['n_selected'])} MOST unusual rows (highest `_anomaly_score` = "
        f"most unusual). They are extremes by construction.\n"
        f"{tier_note}"
        f"Rows of the two groups are interleaved, so the table is not ordered by score. Neither group is a random "
        f"sample of all rows: do not infer trends, averages, shares or totals from these rows, and do not describe the "
        f"extremes as typical. Descriptive statistics computed on all rows appear in the Summary Statistics section, "
        f"when present."
    )