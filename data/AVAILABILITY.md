# Data Availability

This repository intentionally ships **code and key result tables only**. The input data are protected as follows:

## 1. Raw LC-MS metabolomics data (feature matrix `exp_download_express.csv`)

- Non-targeted LC-MS (positive/negative mode, reverse phase) fecal metabolomics of 84 samples.
- Submitted to **MetaboLights** under submission identifier **MJ20240416348** (`Non-targeted LC-MS metabolomics of human feces in cognitive impairment cohorts (AD, MCI, SCD) and healthy controls`).
- The submission is scheduled for public release upon manuscript acceptance; the final accession (MTBLSxxxx) will be updated in this file and in the README.

## 2. Clinical data (CRF table: age, sex, disease duration, MMSE/HAMD/HAMA)

- **Restricted for ethical/privacy reasons** (ethics approval **JAMHC-IRB-202201**, Shanghai Jing'an District Mental Health Center; informed consent obtained from all participants).
- Available from the corresponding author (Gentao Liu, Shanghai Cellpeutics Biotech Co. Ltd.) upon reasonable request, under the terms of the ethics approval.

## 3. Processed results

- All result tables needed to reproduce the manuscript figures and numbers are included under `results/` (Analysis_Results, Tables, Data_Verification).

## Reproduction note

The analysis scripts contain absolute paths from the original analysis machine
(e.g. `C:\Users\PC\Documents\非靶向代谢组学分析\...`) and use the original group label
**CG** ("control group") for the control arm; the manuscript uniformly reports this arm as
**HC** (healthy controls). Adjust the paths to your local layout and the data obtained via
the routes above before running.
