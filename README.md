# STEM–RF–NMF Metabolomics Pipeline for Confounder-Aware Biomarker Discovery in the Alzheimer's Disease Continuum

This repository provides the full Python implementation behind the manuscript *"A reusable confounder-aware metabolomics pipeline integrating STEM, age-residualized Random Forest and NMF subtyping: application to the Alzheimer's disease continuum"* (submitted to *Briefings in Bioinformatics*).

Non-targeted LC-MS metabolomics of **84 human fecal samples** (HC n=10, SCD n=22, MCI n=23, AD n=29) were analyzed with a three-stage pipeline: (1) STEM-based progressive-pattern discovery adapted to ordinal disease stages, (2) age-residualized Random Forest biomarker selection with pharmaceutical/xenobiotic filtering, and (3) NMF molecular subtyping. All numbers below match the manuscript.

## Contents

| Path | What it contains |
|---|---|
| `scripts/` | Analysis pipeline, numbered in execution order (data_processing, analysis1–14, figure generation, verification) |
| `results/` | Key result tables and figures (CSV/JSON/PNG), including the ablation ladder and sensitivity analyses |
| `data/AVAILABILITY.md` | How to obtain the raw / restricted data |
| `requirements.txt` | Software and package versions used |

## Repository layout / workflow

```
data_processing.py            # CRF processing; Table 1 demographics + key metabolite stats
convert_crf.py                # Clinical CRF workbook (.xls) -> crf.csv (encoding used by analysis10-14)
analysis_utils.py             # Shared helpers (STEM ordering, residualization, metrics)
analysis1_permutation_bootstrap.py  # Permutation test (p=0.002) and bootstrap CI [34.5%, 54.8%]
analysis2_sensitivity.py      # Age-correction method comparison (residualization/ComBat/binning)
analysis3_counterfactual.py   # Uncorrected vs corrected: accuracy, xenobiotic proportion, NMF age split
analysis4_classifier_comparison.py  # Five classifiers (GB highest 53.1%; RF 48.8% retained)
analysis5_interpretability.py # MDG vs permutation importance (Spearman rho=0.088)
analysis6_feature_selection.py      # All-features vs ANOVA-500/RFE-200/LASSO (Jaccard stability)
analysis7_nested_cv.py        # Nested CV (50.0%) vs regular CV (48.8%)
analysis8_nmf_clustering.py   # NMF k=2 subtyping (59/25; sil 0.051, cophenetic 0.732) + clustering ARI
analysis9_clinical_correlation.py   # Metabolite-clinical correlation (no FDR-significant hits)
analysis10_ablation.py        # Component-wise ablation: MSI-level filter x age residualization, 2x2 design
analysis11_age_correction_rep10.py  # Four age-correction arms (none/linear/ComBat/binning) under 5x10-fold
analysis13_single_method.py   # Single-method baselines: all features vs ANOVA top-50 / top-200
analysis14_paired_feature_selection.py  # Fold-wise paired test of top-50 vs all-features (50 splits)
generate_main_figures_v2.py   # Main figures Figure 1-6
generate_supplementary_figures.py  # Supplementary figures S1-S10
verify_core_numbers.py        # Cross-checks headline numbers against results/
```
## Reproduce

1. Obtain the data (see `data/AVAILABILITY.md`). Place the feature matrix `exp_download_express.csv` and the clinical CRF workbook in `data/`, then run `python scripts/convert_crf.py` to generate `data/crf.csv`.
2. Paths are resolved relative to the repository root (`ROOT/data` for inputs, `ROOT/results` for outputs), so no editing is required; set `METABO_DATA_DIR` / `METABO_OUT_DIR` only if you keep the data elsewhere.
3. Install dependencies and run the scripts in the order listed above.
```bash
pip install -r requirements.txt
python scripts/data_processing.py
python scripts/convert_crf.py
python scripts/analysis1_permutation_bootstrap.py
# ... analysis2-9 in order, then:
python scripts/analysis10_ablation.py          # 4 configurations, 5x10-fold, 1,000 trees
python scripts/analysis11_age_correction_rep10.py
python scripts/analysis13_single_method.py
python scripts/analysis14_paired_feature_selection.py
python scripts/generate_main_figures_v2.py
python scripts/generate_supplementary_figures.py
python scripts/verify_core_numbers.py
```
- Runtime: minutes on a standard desktop for analysis1-9; analysis10-14 re-run repeated cross-validation with 1,000 trees and take roughly 20-40 min together.
- Random seeds are fixed inside the scripts (global seed 42) for reproducibility.

## Data sources

- **Raw LC-MS data**: submitted to MetaboLights under submission identifier **MJ20240416348** (to be released publicly upon manuscript acceptance; the accession will be updated here).
- **Clinical data**: restricted for ethical/privacy reasons (ethics approval **JAMHC-IRB-202201**); available from the corresponding author upon reasonable request. `data/crf.csv` is derived from that restricted workbook and must not be redistributed.
- **Note on group labels**: the original clinical records and analysis scripts use **CG** ("control group") for the control arm; the manuscript uniformly uses **HC** (healthy controls). They refer to the same subjects.

## Key results reproduced here

- **Component-wise ablation (2x2 design, same 84 cases, 5x10-fold, 1,000 trees)**: age residualization alone accounts for **-11.4 pp** (60.2% -> 48.8%); the MSI-level filter alone accounts for **+0.4 pp** (48.8% -> 49.3%); with both components switched off accuracy is **58.8% (+10.0 pp)**. When age is uncorrected, the top-50 panel contains **0%** B(ii)-level features; opening the filter raises that share to **42-44%** while top-10 cross-repeat Jaccard falls from 0.93 to 0.83.
- Age residualization moves discrimination itself: **kappa 0.44 -> 0.26**, **macro-F1 0.56 -> 0.39** (Supplementary Table S3, Figure 6).
- Three age-correction methods under the same protocol: uncorrected **60.2%**, linear residualization **48.8%**, ComBat **39.3%**, age-binning **40.9%**; AD sensitivity **93/78/71/72%** (Supplementary Figure S2).
- STEM: **2 significant profiles** (MCI-peak n=642, early-rise n=371; P=0.001, FDR=0.025) covering **1,013/1,966** B(i) metabolites.
- Age-residualized RF: four-class accuracy **48.8%** (kappa=0.26); ROC AUC HC/SCD/MCI/AD **0.99/0.61/0.73/0.76**; sensitivity **6/28/50/78%**.
- Single-method baselines: all features **48.8%**, fold-wise ANOVA top-50 **54.3%**, top-200 **46.9%**; the 5.6 pp top-50 gain is significant in a paired test (p=0.004, Wilcoxon p=0.003) but favours top-50 in only 20 of 50 folds. The pipeline therefore trades a quantified 5.6 pp on four-class accuracy for keeping all three outputs on one shared feature space (manuscript section 3.8).
- NMF k=2: subtypes **59/25**; silhouette 0.051, cophenetic 0.732, R2X=0.991.
- Permutation p=0.002; bootstrap 95% CI [34.5%, 54.8%]; nested CV 50.0% vs regular CV 48.8%.
- Cross-method integration: 3-way intersection = 2 metabolites; pairwise 31/42/3.

## Software environment

- Python 3.9, numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, scikit-learn 1.6.1, matplotlib 3.9.4 (conda env `cptac_env`).
- analysis10-14 were re-run and validated with Python 3.13 / scikit-learn 1.9.1; results are identical to the archived JSON/CSV files within rounding.

## Citation

If you use this package, please cite both the manuscript and the archived release:
> Feng Y, Wu R, Kong L, Liu G. A reusable confounder-aware metabolomics pipeline integrating STEM, age-residualized Random Forest and NMF subtyping. *Briefings in Bioinformatics* (under review).
> Code/data archive (Zenodo, concept DOI, always points to the latest version): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23954816.svg)](https://doi.org/10.5281/zenodo.23954816)

## License

MIT; see [LICENSE](LICENSE).

## Contact

Corresponding author: Gentao Liu, Shanghai Cellpeutics Biotech Co. Ltd., Shanghai, China.
