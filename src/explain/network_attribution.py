"""
explain.network_attribution — aggregate per-ROI attribution scores of a
Schaefer parcellation into Yeo functional networks.

The Schaefer parcellations ship with network annotations (Yeo et al. 2011)
encoded in the parcel names of the CBIG label tables
`Schaefer2018_<N>Parcels_{7,17}Networks_order.txt` (columns
`ROI Label, ROI Name, R, A, S`):

    7 networks:   Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default
    17 networks:  VisCent, VisPeri, SomMotA/B, DorsAttnA/B, SalVentAttnA/B,
                  LimbicA/B, ContA/B/C, DefaultA/B/C, TempPar

The atlas volumes used here (`schaefer_<N>_7n_yeo`) number their parcels in
7-network order, so parcel k is row k of the 7-network table. The
17-network table lists the same parcels in a different order; 17-network
names are therefore attached to each parcel by matching parcel centroids
(R, A, S) with the 7-network table.

The label tables are read from SCHAEFER_LABELS_DIR, which defaults to
`src/templates/atlases/labels/` in the anxiety-fmri-classification repository
(see src/paths.py).

Usage:
    from explain.network_attribution import roi_to_network
    network_scores = roi_to_network(
        roi_importance,           # shape (200,), per-ROI score in atlas order
        atlas='schaefer_200_7n_yeo',
        scheme='7networks',       # or '17networks'
    )
    # network_scores: pd.Series, one score per network
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import CH5_SCHAEFER_LABELS_DIR  # noqa: E402

DEFAULT_SCHAEFER_LABELS_DIR = str(CH5_SCHAEFER_LABELS_DIR)


# ---------------------------------------------------------------------------
# Label-table loading
# ---------------------------------------------------------------------------

# CBIG's Schaefer label files have one row per parcel; ROI Name looks like:
#   7-network:  "7Networks_LH_Vis_1", "7Networks_RH_Default_PCC_1"
#   17-network: "17Networks_LH_ContA_PFCl_1", "17Networks_RH_DefaultB_AntTemp_1"
# In both cases the network is the first alpha token after the LH/RH side.
_NETWORK_RE = {
    '7networks':  re.compile(r'7Networks_(?:LH|RH)_(?P<net>[A-Za-z]+)'),
    '17networks': re.compile(r'17Networks_(?:LH|RH)_(?P<net>[A-Za-z]+)'),
}


def _resolve_labels_path(atlas: str, labels_dir: str | None,
                          scheme: str = '7networks') -> str:
    """Find the Schaefer label table for `atlas` and `scheme`.

    scheme: '7networks' (default) or '17networks'.
    """
    labels_dir = labels_dir or os.environ.get(
        'SCHAEFER_LABELS_DIR', DEFAULT_SCHAEFER_LABELS_DIR
    )
    # The atlas name encodes the parcel count; either scheme's label table
    # can be used with the same atlas volume.
    m = re.search(r'schaefer_(\d+)_(?:7|17)n', atlas, re.I)
    if not m:
        raise ValueError(
            f"Atlas '{atlas}' is not a recognised Schaefer name; expected "
            f"e.g. 'schaefer_200_7n_yeo' or 'schaefer_400_7n_yeo'."
        )
    n_rois = int(m.group(1))
    nlabel = '7' if scheme == '7networks' else '17'
    fname  = f'Schaefer2018_{n_rois}Parcels_{nlabel}Networks_order.txt'
    path   = os.path.join(labels_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Schaefer label table not found at {path}. Set ANXIETY_FMRI_REPO "
            f"to the anxiety-fmri-classification checkout, or SCHAEFER_LABELS_DIR "
            f"to a directory holding the CBIG label tables."
        )
    return path


def _read_label_table(path: str) -> pd.DataFrame:
    """Read a CBIG label table into columns roi_index, name (+ R, A, S)."""
    raw = pd.read_csv(path)
    # Tolerate either capitalisation / spacing in the header.
    id_col   = next((c for c in raw.columns if c.lower().strip() in ('roi label', 'roi_label', 'index')), None)
    name_col = next((c for c in raw.columns if c.lower().strip() in ('roi name', 'roi_name', 'name')), None)
    if id_col is None or name_col is None:
        raise ValueError(
            f"Could not find 'ROI Label' / 'ROI Name' columns in {path}; "
            f"got {list(raw.columns)}"
        )
    out = pd.DataFrame({
        'roi_index': raw[id_col].astype(int),
        'name':      raw[name_col].astype(str),
    })
    for c in ('R', 'A', 'S'):
        if c in raw.columns:
            out[c] = raw[c]
    return out


def schaefer_label_table(atlas: str, labels_dir: str | None = None) -> pd.DataFrame:
    """Per-parcel 7- and 17-network names in atlas (7-network) order.

    Returns columns roi_index, name_7, name_17, R, A, S. The 17-network
    name of each parcel is found by matching its centroid (R, A, S) in the
    17-network table. Raises ValueError if any parcel has no unique match.
    """
    t7 = _read_label_table(_resolve_labels_path(atlas, labels_dir, '7networks'))
    t17 = _read_label_table(_resolve_labels_path(atlas, labels_dir, '17networks'))
    xyz = ['R', 'A', 'S']
    if not (set(xyz) <= set(t7.columns) and set(xyz) <= set(t17.columns)):
        raise ValueError('Schaefer label tables need R, A, S centroid columns '
                         'to align the 17-network names with the atlas order.')
    t = t7.merge(t17[['name'] + xyz], on=xyz, how='left', suffixes=('_7', '_17'))
    if len(t) != len(t7) or t['name_17'].isna().any():
        raise ValueError(f"Could not match every parcel of '{atlas}' to a "
                         f"unique 17-network label by centroid.")
    return t[['roi_index', 'name_7', 'name_17'] + xyz]


def load_network_assignment(atlas: str,
                            labels_dir: str | None = None,
                            scheme: str = '7networks') -> pd.DataFrame:
    """Return a DataFrame with columns roi_index (1..N), name, network.

    Rows follow the atlas label order (roi_index = atlas label value).
    scheme: '7networks' (default) or '17networks'. For 17 networks the
    names come from schaefer_label_table (centroid match).
    """
    if scheme == '17networks':
        t = schaefer_label_table(atlas, labels_dir)
        out = pd.DataFrame({'roi_index': t['roi_index'], 'name': t['name_17']})
    elif scheme == '7networks':
        t = _read_label_table(_resolve_labels_path(atlas, labels_dir, scheme))
        out = t[['roi_index', 'name']].copy()
    else:
        raise ValueError(f"Unknown scheme '{scheme}'; use '7networks' or '17networks'.")
    regex = _NETWORK_RE[scheme]
    out['network'] = out['name'].apply(
        lambda n: (m := regex.match(n)) and m.group('net') or 'Unknown'
    )
    return out.sort_values('roi_index').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def roi_to_network(roi_importance: np.ndarray,
                   atlas: str,
                   labels_dir: str | None = None,
                   aggregator: str = 'mean_abs',
                   scheme: str = '7networks') -> pd.Series:
    """Aggregate per-ROI attribution into Yeo network scores.

    Parameters
    ----------
    roi_importance : (n_rois,) array of per-ROI scores in atlas label order,
                     i.e. roi_importance[0] is the parcel with label 1.
    atlas          : 'schaefer_<N>_7n_yeo' atlas name.
    aggregator     : 'mean_abs' (default), 'mean', 'max_abs', or 'sum_abs'.
                     mean_abs averages magnitudes, so the sign of signed
                     attributions does not cancel within a network.
    scheme         : '7networks' (default) or '17networks'.

    Returns
    -------
    pd.Series of network → aggregated score, indexed by network name.
    """
    assignment = load_network_assignment(atlas, labels_dir, scheme=scheme)
    if len(assignment) != len(roi_importance):
        raise ValueError(
            f"ROI count mismatch: atlas has {len(assignment)} parcels, "
            f"roi_importance has {len(roi_importance)} elements."
        )
    df = assignment.copy()
    df['score'] = np.asarray(roi_importance)

    if aggregator == 'mean_abs':
        out = df.groupby('network')['score'].apply(lambda s: np.abs(s).mean())
    elif aggregator == 'mean':
        out = df.groupby('network')['score'].mean()
    elif aggregator == 'max_abs':
        out = df.groupby('network')['score'].apply(lambda s: np.abs(s).max())
    elif aggregator == 'sum_abs':
        out = df.groupby('network')['score'].apply(lambda s: np.abs(s).sum())
    else:
        raise ValueError(f"Unknown aggregator '{aggregator}'")

    return out.sort_index()


def per_network_summary(roi_importance: np.ndarray,
                        atlas: str,
                        labels_dir: str | None = None,
                        scheme: str = '7networks') -> pd.DataFrame:
    """Per-network summary: n_rois in network, mean / std / max of |score|."""
    assignment = load_network_assignment(atlas, labels_dir, scheme=scheme)
    if len(assignment) != len(roi_importance):
        raise ValueError(
            f"ROI count mismatch: atlas {len(assignment)} vs importance "
            f"{len(roi_importance)}."
        )
    df = assignment.copy()
    df['score']     = np.asarray(roi_importance)
    df['abs_score'] = df['score'].abs()
    out = (
        df.groupby('network')
          .agg(n_rois=('score', 'count'),
               mean_signed=('score', 'mean'),
               mean_abs=('abs_score', 'mean'),
               std_abs=('abs_score', 'std'),
               max_abs=('abs_score', 'max'))
          .reset_index()
          .sort_values('mean_abs', ascending=False)
    )
    return out


if __name__ == '__main__':
    # Smoke test with random scores.
    rng = np.random.default_rng(0)
    n = 200
    fake = rng.standard_normal(n)
    print(roi_to_network(fake, 'schaefer_200_7n_yeo'))
    print()
    print(roi_to_network(fake, 'schaefer_200_7n_yeo', scheme='17networks'))
    print()
    print(per_network_summary(fake, 'schaefer_200_7n_yeo'))
