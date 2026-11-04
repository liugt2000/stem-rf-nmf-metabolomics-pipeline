# STEM–RF–NMF Metabolomics Pipeline for Confounder-Aware Biomarker Discovery in the Alzheimer's Disease Continuum

This repository provides the full Python implementation behind the manuscript *"A reusable confounder-aware metabolomics pipeline integrating STEM, age-residualized Random Forest and NMF subtyping: application to the Alzheimer's disease continuum"* (submitted to *Briefings in Bioinformatics*).

Non-targeted LC-MS metabolomics of **84 human fecal samples** (HC n=10, SCD n=22, MCI n=23, AD n=29) were analyzed with a three-stage pipeline: (1) STEM-based progressive-pattern discovery adapted to ordinal disease stages, (2) age-residualized Random Forest biomarker selection with pharmaceutical/xenobiotic filtering, and (3) NMF molecular subtyping. All numbers below match the manuscript.

## Contents

| Path | What it contains |
|---|---|
| `scripts/` | Analysis pipeline, numbered in execution order (analysis1–9, data_processing, figure generation, verification) |
| `results/` | Key result tables as CSV (Analysis_Results, Tables, Data_Verification) |
| `data/AVAILABILITY.md` | How to obtain the raw / restricted data |
| `requirements.txt` | Software and package versions used |

## Repository layout / workflow

```
data_processing.py            # CRF processing; Table 1 demographics + key metabolite stats
analysis_utils.py             # Shared helpers (STEM ordering, residualization, metrics)
analysis1_permutation_bootstrap.py  # Permutation test (p=0.002) and bootstrap CI [34.5%, 54.8%]
analysis2_sensitivity.py      # Age-correction method comparison (residualization/ComBat/binning)
analysis3_counterfactual.py   # Uncorrected vs corrected: accuracy, xenobiotic proportion, NMF age split
analysis4_classifier_comparison.py  # Five classifiers (GB highest 53.1%; RF 48.8% retained)
analysis5_interpretability.py # MDG vs permutation importance (Spearman ρ=0.088)
analysis6_feature_selection.py      # All-features vs ANOVA-500/RFE-200/LASSO (Jaccard stability)
analysis7_nested_cv.py        # Nested CV (50.0%) vs regular CV (48.8%)
analysis8_nmf_clustering.py   # NMF k=2 subtyping (59/25; sil 0.051, cophenetic 0.732) + clustering ARI
analysis9_clinical_correlation.py   # Metabolite–clinical correlation (no FDR-significant hits)
generate_main_figures_v2.py   # Main figures Figure 1–6
generate_supplementary_figures.py  # Supplementary figures S1–S10
verify_core_numbers.py        # Cross-checks headline numbers against results/
```

## Reproduce

1. Obtain the data (see `data/AVAILABILITY.md`) and place it locally.
2. **Adjust the hard-coded absolute input/output paths** at the top of each script (they currently point to the original analysis machine) to your local layout: `data/raw/` for the feature matrix `exp_download_express.csv` and clinical CRF table, `results/` for outputs.
3. Install dependencies and run the scripts in the order listed above.

```bash
pip install -r requirements.txt
python scripts/data_processing.py
python scripts/analysis1_permutation_bootstrap.py
# ... analysis2–9 in order, then:
python scripts/generate_main_figures_v2.py
python scripts/generate_supplementary_figures.py
python scripts/verify_core_numbers.py
```

- Runtime: minutes on a standard desktop (n=84; cross-validation and permutations are the slowest steps).
- Random seeds are fixed inside the scripts for reproducibility.

## Data sources

- **Raw LC-MS data**: submitted to MetaboLights under submission identifier **MJ20240416348** (to be released publicly upon manuscript acceptance; the accession will be updated here).
- **Clinical data**: restricted for ethical/privacy reasons (ethics approval **JAMHC-IRB-202201**); available from the corresponding author upon reasonable request.
- **Note on group labels**: the original clinical records and analysis scripts use **CG** ("control group") for the control arm; the manuscript uniformly uses **HC** (healthy controls). They refer to the same subjects.

## Key results reproduced here

- Two-stage accuracy decomposition: **60.47% → 48.8% (−11.7 pp**, demographic confounding; chance 25%).
- STEM: **2 significant profiles** (MCI-peak n=642, early-rise n=371; P=0.001, FDR=0.025) covering **1,013/1,966** B(i) metabolites.
- Age-residualized RF: four-class accuracy **48.8%** (κ=0.26); ROC AUC HC/SCD/MCI/AD **0.99/0.61/0.73/0.76**; sensitivity **6/28/50/78%**.
- NMF k=2: subtypes **59/25**; silhouette 0.051, cophenetic 0.732, R²X=0.991.
- Permutation p=0.002; bootstrap 95% CI [34.5%, 54.8%]; nested CV 50.0% vs regular CV 48.8%.
- Cross-method integration: 3-way intersection = 2 metabolites; pairwise 31/42/3.

## Software environment

- Python 3.9, numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, scikit-learn 1.6.1, matplotlib 3.9.4 (conda env `cptac_env`).

## Citation

If you use this package, please cite both the manuscript and the archived release:

> Feng Y, Wu R, Kong L, Liu G. A reusable confounder-aware metabolomics pipeline integrating STEM, age-residualized Random Forest and NMF subtyping. *Briefings in Bioinformatics* (under review).
> Code/data archive (Zenodo, concept DOI, always points to the latest version): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23954816.svg)](https://doi.org/10.5281/zenodo.23954816)

## License

MIT; see [LICENSE](LICENSE).

## Contact

Corresponding author: Gentao Liu, Shanghai Cellpeutics Biotech Co. Ltd., Shanghai, China.
