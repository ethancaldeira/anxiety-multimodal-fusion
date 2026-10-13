"""
Chapter 7 explainability helpers, imported by src/report_saliency.py and
src/llm_report.py:

    methods               attribution methods with a uniform API
                          (Grad-CAM, Grad-CAM++, SmoothGrad, Integrated
                          Gradients).
    network_attribution   per-ROI scores of a Schaefer parcellation →
                          Yeo 7- or 17-network scores, using the CBIG label
                          tables from the anxiety-fmri-classification repository.
    reporting             sorted horizontal bar chart of network scores.
    llm_narrative         evidence collection, deterministic reliability
                          gate and prompt construction for the optional
                          LLM report.
    atlas_glossary        plain-anatomy names for AAL regions and
                          Schaefer / Yeo codes.
"""
