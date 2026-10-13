# Explainability helpers

Support modules for `src/report_saliency.py` and `src/llm_report.py`; the
command-line entry points are one level up.

| File | Purpose | Reference |
|---|---|---|
| `methods.py`              | Attribution methods with a uniform API: Grad-CAM and Grad-CAM++ (implemented here, including nested backbones such as ResNet50), SmoothGrad (tf-keras-vis) and Integrated Gradients (implemented here; all-zeros baseline, 50 steps by default). | Selvaraju 2017 (Grad-CAM); Chattopadhay 2018 (Grad-CAM++); Smilkov 2017 (SmoothGrad); Sundararajan 2017 (Integrated Gradients) |
| `network_attribution.py`  | Per-ROI scores of a Schaefer parcellation → Yeo 7-network scores (Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default) or 17-network scores. The 17-network names are matched to the atlas parcels by centroid. | Schaefer 2018; Yeo 2011 |
| `reporting.py`            | Sorted horizontal bar chart of network scores. | — |
| `llm_narrative.py`        | Evidence collection → deterministic reliability gate (bootstrap AUC CI; fixed-mean fusion checks) → prompt for the optional LLM report. Used by `src/llm_report.py`. | — |
| `atlas_glossary.py`       | Plain-anatomy names for AAL regions and Schaefer/Yeo parcel and network codes (e.g. `pCun` → precuneus), attached to the LLM evidence so the model does not expand atlas codes itself. | Schaefer 2018; Yeo 2011; Tzourio-Mazoyer 2002 |
| `labels/aal_spm12.csv`    | AAL code → region name and hemisphere (116 regions, SPM12 AAL label table), for labelling AAL ROI scores. | Tzourio-Mazoyer 2002 |

The Schaefer label tables (`Schaefer2018_<N>Parcels_{7,17}Networks_order.txt`,
CBIG) are read from the anxiety-fmri-classification repository
(`$ANXIETY_FMRI_REPO/src/templates/atlases/labels/`), or from
`SCHAEFER_LABELS_DIR` if set.
