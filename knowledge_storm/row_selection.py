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
  - Returns only the top K most anomalous rows (default: 25), SORTED by
    descending score, including the `_anomaly_score` and a diagnostic dictionary.
  - Small tables (<= `full_threshold` rows) are passed through unchanged.
"""

import logging
import math
import re
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
_HDBSCAN_MAX_ROWS = 20000
_MIN_GROUP_SIZE = 20
_Z_CLIP = 20.0


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
    scale = 0.0
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
) -> Tuple[Optional[np.ndarray], List[str], Dict[str, str]]:
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
) -> Tuple[np.ndarray, str]:
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
    rows: Optional[List[Dict[str, Any]]],
    *,
    full_threshold: int = 50,
    top_frac: float = 0.01,
    min_rows: int = 10,
    max_rows: int = 25,
    exclude_columns: Sequence[str] = (),
    group_by: Sequence[str] = (),
    random_state: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (rows_to_show, info). It never raises an exception: in the event of an error or unusable features,
    it returns the original rows.
    """
    rows = rows or []
    n = len(rows)
    info: Dict[str, Any] = {
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
        k = min(n, max_rows, max(min_rows, math.ceil(top_frac * n)))
        top_idx = np.argsort(-scores, kind="stable")[:k]  # descending score: token truncation discards the least anomalous ones

        selected = [{**rows[i], "_anomaly_score": round(float(scores[i]), 3)} for i in top_idx]
        info.update(n_selected=len(selected), features_used=used)
        return selected, info
    except Exception as e:  # noqa: BLE001
        logger.warning("selection of anomalous rows failed, using all rows: %s", e)
        info["error"] = str(e)
        return rows, info


def describe_selection(info: Dict[str, Any]) -> str:
    """Note to precede the table shown in the template. Empty string if no selection was made."""
    if info.get("n_selected", 0) >= info.get("n_total", 0):
        return ""
    used = ", ".join(info.get("features_used", [])) or "n/a"
    grouping = (
        f", relative to each {'/'.join(info['group_by'])}" if info.get("group_by") else ""
    )
    return (
        f"**Note:** the query returned {info['n_total']} rows. Only the {info['n_selected']} most "
        f"anomalous rows are shown below ( measures: {used}{grouping}), "
        f"sorted by `_anomaly_score`, most unusual first (higher = more unusual). This is NOT a representative sample: "
        f"do not infer trends, averages or totals from these rows. Descriptive statistics computed on "
        f"all rows appear in the Summary Statistics section, when present."
    )