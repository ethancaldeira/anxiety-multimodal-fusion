"""
paths.py — locations of the two upstream thesis repositories.

The Chapter 7 code reads trained models, predictions, cached connectivity
features, atlases and slice images from the repositories that train the two
branches:

    ANXIETY_SMRI_REPO   anxiety-smri-classification (structural branch)
    ANXIETY_FMRI_REPO   anxiety-fmri-classification (functional branch)

Each environment variable defaults to a sibling directory of this
repository, so cloning the repositories side by side needs no
configuration:

    <parent>/
    ├── anxiety-smri-classification/
    ├── anxiety-fmri-classification/
    └── anxiety-multimodal-fusion/   (this repository)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SIBLING_ROOT = Path(__file__).resolve().parents[2]


def _repo_path(env_var: str, sibling_name: str) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser() if value else _SIBLING_ROOT / sibling_name


CH4_REPO = _repo_path('ANXIETY_SMRI_REPO', 'anxiety-smri-classification')
CH5_REPO = _repo_path('ANXIETY_FMRI_REPO', 'anxiety-fmri-classification')

# Locations inside the functional-branch repository.
CH5_ATLAS_DIR = CH5_REPO / 'src' / 'templates' / 'atlases'
CH5_CACHE_DIR = CH5_REPO / 'data' / 'cache'
CH5_SCHAEFER_LABELS_DIR = CH5_ATLAS_DIR / 'labels'

# Default results directory of this repository.
DEFAULT_RESULTS_DIR = REPO_ROOT / 'results' / 'ch7_v2'
