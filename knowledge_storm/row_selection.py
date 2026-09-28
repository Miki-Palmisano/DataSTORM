"""
Selezione delle righe anomale di un risultato SQL.

Posizione consigliata nel repo: knowledge_storm/row_selection.py
(modulo foglia: non importa nulla da knowledge_storm).

Cosa fa
  - prende il risultato completo (list[dict], come execution_result_full_dict)
  - usa come feature SOLO le colonne numeriche (misure), con scaling robusto
    (mediana/IQR, log1p per i conteggi molto asimmetrici). Le colonne categoriche
    NON entrano nelle feature: restano come contesto nelle righe restituite e,
    se indicate in `group_by`, definiscono "anomalo rispetto al proprio gruppo"
    (es. group_by=["region"]: 20 eventi sono normali a Bogotà, anomali in un
    dipartimento con media 3)
  - esclude colonne temporali, id e costanti (il tempo sarà trattato dalle
    finestre temporali, non dal clustering)
  - assegna a ogni riga uno score di anomalia con IsolationForest (default, lineare, validato
    nei test sintetici). HDBSCAN/GLOSH è disponibile solo su richiesta esplicita
    (method="hdbscan", n <= 20000, richiede il pacchetto `hdbscan`) e NON è validato
  - restituisce solo le K righe più anomale (con `_anomaly_score`) + un dict diagnostico
  - tabelle piccole (<= full_threshold righe) passano invariate

Perché niente one-hot: in un test sintetico (30k righe, 15 anomalie iniettate) l'aggiunta
delle categoriche in one-hot faceva recuperare 1 anomalia su 15, perché 15 colonne binarie
su 18 dominano le suddivisioni casuali dell'IsolationForest e diluiscono il segnale delle misure.

Regola d'uso: le statistiche descrittive vanno calcolate SEMPRE sul risultato completo,
indipendentemente da questa selezione. Le righe selezionate NON sono un campione
rappresentativo: describe_selection() produce la nota da mostrare al modello.
"""

import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_TIME_HINTS = ("week", "date", "time", "year", "month", "period")
_HDBSCAN_MAX_ROWS = 20000
_MIN_GROUP_SIZE = 20
_Z_CLIP = 20.0


def _looks_temporal(name: str, series) -> bool:
    if any(h in name.lower() for h in _TIME_HINTS):
        return True
    sample = series.dropna().astype(str).head(20)
    return len(sample) > 0 and all(_DATE_RE.match(s) for s in sample)


def _is_id_like(name: str, series, n: int) -> bool:
    import pandas as pd

    lname = name.lower()
    if lname == "id" or lname.endswith("_id") or lname.endswith("_cnty"):
        return True
    if n >= 50 and series.nunique(dropna=True) / n > 0.95:
        # tutti valori distinti: identificatore, tranne i float continui (misure)
        return not pd.api.types.is_float_dtype(series)
    return False


def _robust_z(v: np.ndarray) -> np.ndarray:
    """(v - mediana) / scala robusta; la scala scende da IQR a intervalli più larghi se i dati sono quasi tutti uguali."""
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
    """Ritorna (X, colonne_numeriche_usate, {colonna_scartata: motivo})."""
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
        # asimmetria calcolata sul "corpo" della distribuzione (p1-p99): gli outlier stessi
        # non devono decidere la trasformazione, altrimenti log1p ne comprime il segnale
        lo, hi = np.percentile(v, [1, 99])
        skew = pd.Series(np.clip(v, lo, hi)).skew()
        if v.min() >= 0 and not np.isnan(skew) and skew > 2:
            v = np.log1p(v)  # conteggi molto asimmetrici

        if groups is None:
            z = _robust_z(v)
        else:
            z = np.full(n, np.nan)
            for gid in np.unique(groups):
                m = groups == gid
                if m.sum() >= _MIN_GROUP_SIZE:
                    z[m] = _robust_z(v[m])
            missing = np.isnan(z)  # gruppi troppo piccoli: statistiche globali
            if missing.any():
                z[missing] = _robust_z(v)[missing]

        cols.append(z)
        used.append(col)

    if not cols:
        return None, [], dropped
    return np.column_stack(cols), used, dropped


def score_anomalies(
    X: np.ndarray, method: str = "auto", random_state: int = 0
) -> Tuple[np.ndarray, str]:
    """Score di anomalia per riga (più alto = più anomalo) e nome del metodo effettivamente usato."""
    n = len(X)
    if method == "auto":
        # "auto" non sceglie metodi non validati: HDBSCAN solo se richiesto esplicitamente
        method = "isolation_forest"
    if method == "hdbscan" and n > _HDBSCAN_MAX_ROWS:
        logger.warning("HDBSCAN limitato a %d righe (n=%d): uso IsolationForest", _HDBSCAN_MAX_ROWS, n)
        method = "isolation_forest"

    if method == "hdbscan":
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(5, int(0.01 * n)), core_dist_n_jobs=-1
            ).fit(X)
            return np.nan_to_num(clusterer.outlier_scores_, nan=0.0), "hdbscan"
        except Exception as e:  # noqa: BLE001  (ImportError o dati non adatti)
            logger.warning("HDBSCAN non utilizzabile (%s): uso IsolationForest", e)

    from sklearn.ensemble import IsolationForest

    # max_samples alto: con il default (256) un'anomalia rara compare in pochi alberi, e su una
    # misura quasi costante gli alberi senza anomalie non hanno alcuno split su quella misura
    iso = IsolationForest(
        n_estimators=100, max_samples=min(n, 10000), random_state=random_state, n_jobs=-1
    ).fit(X)
    return -iso.score_samples(X), "isolation_forest"


def select_anomalous_rows(
    rows: Optional[List[Dict[str, Any]]],
    *,
    full_threshold: int = 50,
    top_frac: float = 0.01,
    min_rows: int = 10,
    max_rows: int = 50,
    method: str = "auto",
    exclude_columns: Sequence[str] = (),
    group_by: Sequence[str] = (),
    random_state: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Ritorna (righe_da_mostrare, info). Non solleva mai: in caso di errore o feature
    non utilizzabili ritorna le righe originali.
    """
    rows = rows or []
    n = len(rows)
    info: Dict[str, Any] = {
        "n_total": n,
        "n_selected": n,
        "method": None,
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

        scores, used_method = score_anomalies(X, method=method, random_state=random_state)
        k = min(n, max_rows, max(min_rows, math.ceil(top_frac * n)))
        top_idx = np.sort(np.argsort(-scores)[:k])  # ordine originale (utile se già ordinato per tempo)

        selected = [{**rows[i], "_anomaly_score": round(float(scores[i]), 3)} for i in top_idx]
        info.update(n_selected=len(selected), method=used_method, features_used=used)
        return selected, info
    except Exception as e:  # noqa: BLE001
        logger.warning("selezione righe anomale fallita, uso tutte le righe: %s", e)
        info["error"] = str(e)
        return rows, info


def describe_selection(info: Dict[str, Any]) -> str:
    """Nota da anteporre alla tabella mostrata al modello. Stringa vuota se non c'è stata selezione."""
    if info.get("n_selected", 0) >= info.get("n_total", 0):
        return ""
    used = ", ".join(info.get("features_used", [])) or "n/a"
    grouping = (
        f", relative to each {'/'.join(info['group_by'])}" if info.get("group_by") else ""
    )
    return (
        f"**Note:** the query returned {info['n_total']} rows. Only the {info['n_selected']} most "
        f"anomalous rows are shown below (method: {info['method']}; measures: {used}{grouping}), "
        f"ranked by `_anomaly_score` (higher = more unusual). This is NOT a representative sample: "
        f"do not infer trends, averages or totals from these rows. Descriptive statistics computed on "
        f"all rows appear in the Summary Statistics section, when present."
    )