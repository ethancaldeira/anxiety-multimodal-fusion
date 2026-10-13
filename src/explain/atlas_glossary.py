"""
explain.atlas_glossary — plain-anatomy names for AAL and Schaefer/Yeo labels.

The LLM narrative does not expand atlas codes itself: code attaches the
plain-anatomy name (for example Schaefer `pCun` -> precuneus), and an
unknown code yields None so nothing is guessed.

Sources: Schaefer et al. 2018 / Kong et al. 2019 label conventions (CBIG
Schaefer2018_*Parcels_{7,17}Networks_order files); Yeo et al. 2011 network
names; Tzourio-Mazoyer et al. 2002 AAL region names.
"""
from __future__ import annotations

import re

YEO_NETWORKS = {
    # 7-network
    'Vis':          'visual network',
    'SomMot':       'somatomotor network',
    'DorsAttn':     'dorsal attention network',
    'SalVentAttn':  'salience / ventral attention network',
    'Limbic':       'limbic network',
    'Cont':         'frontoparietal control network',
    'Default':      'default mode network',
    # 17-network
    'VisCent':      'central visual network',
    'VisPeri':      'peripheral visual network',
    'SomMotA':      'somatomotor network A',
    'SomMotB':      'somatomotor network B',
    'DorsAttnA':    'dorsal attention network A',
    'DorsAttnB':    'dorsal attention network B',
    'SalVentAttnA': 'salience / ventral attention network A',
    'SalVentAttnB': 'salience / ventral attention network B',
    'LimbicA':      'limbic network A',
    'LimbicB':      'limbic network B',
    'ContA':        'frontoparietal control network A',
    'ContB':        'frontoparietal control network B',
    'ContC':        'frontoparietal control network C',
    'DefaultA':     'default mode network A',
    'DefaultB':     'default mode network B',
    'DefaultC':     'default mode network C',
    'TempPar':      'temporo-parietal network',
}

# Sub-region codes that follow the network in Schaefer parcel names.
SCHAEFER_REGIONS = {
    'AntTemp':    'anterior temporal cortex',
    'Aud':        'auditory cortex',
    'Cent':       'central (pre/postcentral) cortex',
    'Cing':       'cingulate cortex',
    'Cingm':      'mid-cingulate cortex',
    'Cingp':      'posterior cingulate cortex',
    'ExStr':      'extrastriate visual cortex',
    'ExStrInf':   'inferior extrastriate visual cortex',
    'ExStrSup':   'superior extrastriate visual cortex',
    'FEF':        'frontal eye field',
    'FrMed':      'medial frontal cortex',
    'FrOper':     'frontal operculum',
    'FrOperIns':  'frontal operculum / insula',
    'IPL':        'inferior parietal lobule',
    'IPS':        'intraparietal sulcus',
    'Ins':        'insula',
    'Med':        'medial cortex',
    'OFC':        'orbitofrontal cortex',
    'PFC':        'prefrontal cortex',
    'PFCd':       'dorsal prefrontal cortex',
    'PFCdPFCm':   'dorsal / medial prefrontal cortex',
    'PFCl':       'lateral prefrontal cortex',
    'PFCld':      'dorsolateral prefrontal cortex',
    'PFClv':      'ventrolateral prefrontal cortex',
    'PFCm':       'medial prefrontal cortex',
    'PFCmp':      'posterior medial prefrontal cortex',
    'PFCv':       'ventral prefrontal cortex',
    'PHC':        'parahippocampal cortex',
    'Par':        'parietal cortex',
    'ParMed':     'medial parietal cortex',
    'ParOcc':     'parieto-occipital cortex',
    'ParOper':    'parietal operculum',
    'Post':       'posterior cortex',
    'PostC':      'postcentral gyrus',
    'PrC':        'precentral gyrus',
    'PrCv':       'ventral precentral gyrus',
    'Rsp':        'retrosplenial cortex',
    'S2':         'secondary somatosensory cortex',
    'SPL':        'superior parietal lobule',
    'StriCal':    'striate / calcarine (primary visual) cortex',
    'Striate':    'striate (primary visual) cortex',
    'Temp':       'temporal cortex',
    'TempOcc':    'temporo-occipital cortex',
    'TempOccPar': 'temporo-occipito-parietal cortex',
    'TempPole':   'temporal pole',
    'pCun':       'precuneus',
    'pCunPCC':    'precuneus / posterior cingulate cortex',
}

AAL_REGIONS = {
    'Precentral':          'precentral gyrus',
    'Frontal_Sup':         'superior frontal gyrus',
    'Frontal_Sup_Orb':     'superior frontal gyrus, orbital part',
    'Frontal_Mid':         'middle frontal gyrus',
    'Frontal_Mid_Orb':     'middle frontal gyrus, orbital part',
    'Frontal_Inf_Oper':    'inferior frontal gyrus, opercular part',
    'Frontal_Inf_Tri':     'inferior frontal gyrus, triangular part',
    'Frontal_Inf_Orb':     'inferior frontal gyrus, orbital part',
    'Rolandic_Oper':       'Rolandic operculum',
    'Supp_Motor_Area':     'supplementary motor area',
    'Olfactory':           'olfactory cortex',
    'Frontal_Sup_Medial':  'medial superior frontal gyrus',
    'Frontal_Med_Orb':     'medial orbitofrontal cortex',
    'Rectus':              'gyrus rectus',
    'Insula':              'insula',
    'Cingulum_Ant':        'anterior cingulate cortex',
    'Cingulum_Mid':        'middle cingulate cortex',
    'Cingulum_Post':       'posterior cingulate cortex',
    'Hippocampus':         'hippocampus',
    'ParaHippocampal':     'parahippocampal gyrus',
    'Amygdala':            'amygdala',
    'Calcarine':           'calcarine cortex (primary visual)',
    'Cuneus':              'cuneus',
    'Lingual':             'lingual gyrus',
    'Occipital_Sup':       'superior occipital gyrus',
    'Occipital_Mid':       'middle occipital gyrus',
    'Occipital_Inf':       'inferior occipital gyrus',
    'Fusiform':            'fusiform gyrus',
    'Postcentral':         'postcentral gyrus',
    'Parietal_Sup':        'superior parietal lobule',
    'Parietal_Inf':        'inferior parietal lobule',
    'SupraMarginal':       'supramarginal gyrus',
    'Angular':             'angular gyrus',
    'Precuneus':           'precuneus',
    'Paracentral_Lobule':  'paracentral lobule',
    'Caudate':             'caudate nucleus',
    'Putamen':             'putamen',
    'Pallidum':            'globus pallidus',
    'Thalamus':            'thalamus',
    'Heschl':              "Heschl's gyrus (primary auditory)",
    'Temporal_Sup':        'superior temporal gyrus',
    'Temporal_Pole_Sup':   'temporal pole, superior part',
    'Temporal_Mid':        'middle temporal gyrus',
    'Temporal_Pole_Mid':   'temporal pole, middle part',
    'Temporal_Inf':        'inferior temporal gyrus',
}

_SIDE = {'L': 'left', 'R': 'right', 'LH': 'left', 'RH': 'right'}
_CEREB_LOBULE = {'Crus1': 'crus I', 'Crus2': 'crus II', '3': 'lobule III',
                 '4_5': 'lobules IV-V', '6': 'lobule VI', '7b': 'lobule VIIb',
                 '8': 'lobule VIII', '9': 'lobule IX', '10': 'lobule X'}
_VERMIS_LOBULE = {'1_2': 'lobules I-II', '3': 'lobule III', '4_5': 'lobules IV-V',
                  '6': 'lobule VI', '7': 'lobule VII', '8': 'lobule VIII',
                  '9': 'lobule IX', '10': 'lobule X'}


def network_name(code: str) -> str | None:
    return YEO_NETWORKS.get(code)


def describe_aal(name: str) -> str | None:
    """'Cingulum_Post_L' -> 'left posterior cingulate cortex'."""
    if name.startswith('Vermis_'):
        lob = _VERMIS_LOBULE.get(name[len('Vermis_'):])
        return f'cerebellar vermis, {lob}' if lob else None
    m = re.fullmatch(r'(.+)_([LR])', name)
    if not m:
        return None
    base, side = m.group(1), _SIDE[m.group(2)]
    if base.startswith('Cerebelum_'):
        lob = _CEREB_LOBULE.get(base[len('Cerebelum_'):])
        return f'{side} cerebellar hemisphere, {lob}' if lob else None
    region = AAL_REGIONS.get(base)
    return f'{side} {region}' if region else None


def describe_schaefer(name: str) -> str | None:
    """'RH_ContC_pCun_2' (17n name, prefix stripped) -> 'right precuneus'.

    Parcels without a sub-region code are described by network only.
    """
    parts = re.sub(r'^(?:7|17)Networks_', '', name).split('_')
    if len(parts) < 2 or parts[0] not in _SIDE or parts[1] not in YEO_NETWORKS:
        return None
    side, net = _SIDE[parts[0]], YEO_NETWORKS[parts[1]]
    regions = [p for p in parts[2:] if not p.isdigit()]
    if not regions:
        return f'{side} hemisphere, {net}'
    if any(r not in SCHAEFER_REGIONS for r in regions):
        return None
    return f"{side} {', '.join(SCHAEFER_REGIONS[r] for r in regions)} ({net})"
