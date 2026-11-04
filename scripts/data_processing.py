# -*- coding: utf-8 -*-
"""
Step 1: Clinical CRF data processing & CSV metabolite verification
Generates: Table 1 (demographics), key metabolite stats, RF biomarker details
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========== 1. Clinical CRF Data Processing ==========
print("=" * 60)
print("Step 1: Processing Clinical CRF Data")
print("=" * 60)

# Read CRF Excel file
crf_path = r"C:\Users\PC\Documents\非靶向代谢组学分析\临床CRF表.xls"
crf_df = pd.read_excel(crf_path)
print(f"CRF shape: {crf_df.shape}")
print(f"Columns: {list(crf_df.columns)}")
print(f"\nFirst 3 rows:\n{crf_df.head(3)}")

# Clean group labels - unify " MCI " and "MCI"
group_col = crf_df.columns[0]  # First column is group
crf_df[group_col] = crf_df[group_col].astype(str).str.strip()
print(f"\nGroup distribution:\n{crf_df[group_col].value_counts()}")

# Define group order
group_order = ['CG', 'SCD', 'MCI', 'AD']

# ========== Generate Table 1: Demographics ==========
print("\n" + "=" * 60)
print("Generating Table 1: Demographic and Clinical Characteristics")
print("=" * 60)

table1_rows = []

for col in crf_df.columns:
    if col == group_col:
        continue

    col_data = crf_df[col]

    # Check if numeric
    try:
        numeric_data = pd.to_numeric(col_data, errors='coerce')
        if numeric_data.notna().sum() > len(col_data) * 0.5:
            # Numeric variable - report mean ± SD
            row_data = {'Variable': col}
            for grp in group_order:
                grp_vals = numeric_data[crf_df[group_col] == grp].dropna()
                if len(grp_vals) > 0:
                    row_data[grp] = f"{grp_vals.mean():.1f} ± {grp_vals.std():.1f}"
                else:
                    row_data[grp] = "N/A"

            # Kruskal-Wallis test (non-parametric for small samples)
            groups_data = []
            for grp in group_order:
                grp_vals = numeric_data[crf_df[group_col] == grp].dropna()
                if len(grp_vals) > 0:
                    groups_data.append(grp_vals.values)

            if len(groups_data) >= 2:
                try:
                    h_stat, p_val = stats.kruskal(*groups_data)
                    row_data['P-value'] = f"{p_val:.3f}" if p_val >= 0.001 else f"<0.001"
                except:
                    row_data['P-value'] = "N/A"
            else:
                row_data['P-value'] = "N/A"

            table1_rows.append(row_data)
        else:
            raise ValueError("Not numeric")
    except:
        # Categorical variable - report n (%)
        categories = col_data.dropna().unique()
        for cat in sorted(categories.astype(str)):
            row_data = {'Variable': f"{col} - {cat}"}
            total_n = len(col_data.dropna())
            for grp in group_order:
                grp_vals = col_data[crf_df[group_col] == grp].dropna()
                grp_total = len(grp_vals)
                cat_count = (grp_vals.astype(str) == str(cat)).sum()
                if grp_total > 0:
                    row_data[grp] = f"{cat_count} ({cat_count/grp_total*100:.1f}%)"
                else:
                    row_data[grp] = "N/A"

            # Chi-square test
            try:
                contingency = []
                for grp in group_order:
                    grp_vals = col_data[crf_df[group_col] == grp].dropna()
                    cat_count = (grp_vals.astype(str) == str(cat)).sum()
                    contingency.append([cat_count, len(grp_vals) - cat_count])
                chi2, p_val, _, _ = stats.chi2_contingency(contingency)
                row_data['P-value'] = f"{p_val:.3f}" if p_val >= 0.001 else f"<0.001"
            except:
                row_data['P-value'] = "N/A"

            table1_rows.append(row_data)

table1_df = pd.DataFrame(table1_rows)
# Reorder columns
table1_df = table1_df[['Variable'] + group_order + ['P-value']]
print(table1_df.to_string(index=False))

# Save Table 1
table1_df.to_csv(r"C:\Users\PC\Documents\非靶向代谢组学分析\Tables\Table_01_Demographics.csv", index=False, encoding='utf-8-sig')
print("\nTable 1 saved to Tables/Table_01_Demographics.csv")

# ========== 2. CSV Metabolite Verification ==========
print("\n" + "=" * 60)
print("Step 2: CSV Metabolite Data Verification")
print("=" * 60)

csv_path = r"C:\Users\PC\Documents\非靶向代谢组学分析\exp_download_express.csv"
meta_df = pd.read_csv(csv_path)
print(f"CSV shape: {meta_df.shape}")
print(f"Columns (first 20): {list(meta_df.columns[:20])}")

# Identify sample columns (exclude annotation columns)
annotation_cols = ['ID', 'Metabolite', 'metab_id', 'Library ID', 'level', 'KEGG Compound ID',
                   'M/Z', 'Retention time', 'Mode', 'Adducts', 'Formula',
                   'Fragmentation Score', 'Theoretical Fragmentation Score', 'Mass Error',
                   'CAS ID', 'RSD']
sample_cols = [c for c in meta_df.columns if c not in annotation_cols]
# Remove QC columns
qc_cols = [c for c in sample_cols if c.startswith('QC')]
exp_sample_cols = [c for c in sample_cols if c not in qc_cols]
print(f"\nAnnotation columns: {len(annotation_cols)}")
print(f"Experimental sample columns: {len(exp_sample_cols)}")
print(f"QC columns: {len(qc_cols)}")

# Classify samples by group
def get_group(col_name):
    col_upper = col_name.upper().strip()
    if col_upper.startswith('AD'):
        return 'AD'
    elif col_upper.startswith('MCI'):
        return 'MCI'
    elif col_upper.startswith('SCD'):
        return 'SCD'
    elif col_upper.startswith('CG'):
        return 'CG'
    return 'Unknown'

sample_groups = {col: get_group(col) for col in exp_sample_cols}
group_counts = pd.Series(sample_groups).value_counts()
print(f"\nSample group distribution: {dict(group_counts)}")

# ========== Key Metabolites Verification ==========
print("\n--- Key Metabolites Verification ---")

key_metabolites = {
    'GABA': 'C00334',
    'L-Serine': 'C00065',
    'L-Carnitine': 'C00318',
    'Citrulline': 'C00327',
    "Adenosine 3'-monophosphate": 'C01367',
    'Deoxyadenosine': 'C00559',
    'Phenethylamine': 'C05332',
    'Xanthurenic Acid': 'C02470',
    'Taurine': 'C00245',
    'Cadaverine': 'C01672',
}

# Find metabolites in CSV by KEGG ID
key_met_stats = []

for met_name, kegg_id in key_metabolites.items():
    # Search by KEGG Compound ID
    matches = meta_df[meta_df['KEGG Compound ID'] == kegg_id]
    if len(matches) == 0:
        # Try searching by name
        matches = meta_df[meta_df['Metabolite'].str.contains(met_name, case=False, na=False)]

    if len(matches) == 0:
        print(f"  WARNING: {met_name} ({kegg_id}) not found in CSV!")
        key_met_stats.append({
            'Metabolite': met_name, 'KEGG_ID': kegg_id, 'Status': 'NOT FOUND'
        })
        continue

    # Take first match
    met_row = matches.iloc[0]
    met_id = met_row['metab_id']
    csv_id = met_row['ID']
    mz = met_row['M/Z']
    rt = met_row['Retention time']
    mode = met_row['Mode']
    level = met_row['level']

    print(f"  Found: {met_name} | ID={csv_id} | metab_id={met_id} | KEGG={kegg_id} | MZ={mz} | RT={rt} | Mode={mode}")

    # Extract expression values by group
    group_values = {}
    for grp in group_order:
        grp_cols = [c for c in exp_sample_cols if sample_groups[c] == grp]
        if len(grp_cols) > 0:
            vals = pd.to_numeric(met_row[grp_cols], errors='coerce').dropna()
            group_values[grp] = vals
        else:
            group_values[grp] = pd.Series(dtype=float)

    # Calculate stats
    stat_row = {
        'Metabolite': met_name,
        'KEGG_ID': kegg_id,
        'CSV_ID': csv_id,
        'metab_id': met_id,
        'MZ': mz,
        'RT': rt,
        'Mode': mode,
        'Level': level,
        'Status': 'FOUND'
    }

    # Group means ± SD
    for grp in group_order:
        vals = group_values[grp]
        if len(vals) > 0:
            stat_row[f'{grp}_mean'] = vals.mean()
            stat_row[f'{grp}_sd'] = vals.std()
            stat_row[f'{grp}_n'] = len(vals)
        else:
            stat_row[f'{grp}_mean'] = np.nan
            stat_row[f'{grp}_sd'] = np.nan
            stat_row[f'{grp}_n'] = 0

    # ANOVA / Kruskal-Wallis test
    groups_data = [group_values[g].values for g in group_order if len(group_values[g]) > 0]
    if len(groups_data) >= 2:
        try:
            f_stat, p_val = stats.f_oneway(*groups_data)
            stat_row['ANOVA_F'] = f_stat
            stat_row['ANOVA_P'] = p_val
        except:
            stat_row['ANOVA_F'] = np.nan
            stat_row['ANOVA_P'] = np.nan

        try:
            h_stat, kw_p = stats.kruskal(*groups_data)
            stat_row['KW_H'] = h_stat
            stat_row['KW_P'] = kw_p
        except:
            stat_row['KW_H'] = np.nan
            stat_row['KW_P'] = np.nan
    else:
        stat_row['ANOVA_F'] = np.nan
        stat_row['ANOVA_P'] = np.nan
        stat_row['KW_H'] = np.nan
        stat_row['KW_P'] = np.nan

    # Fold changes (AD vs CG)
    if len(group_values['AD']) > 0 and len(group_values['CG']) > 0:
        ad_mean = group_values['AD'].mean()
        cg_mean = group_values['CG'].mean()
        if cg_mean > 0:
            stat_row['FC_AD_vs_CG'] = ad_mean / cg_mean
            stat_row['log2FC_AD_vs_CG'] = np.log2(ad_mean / cg_mean) if ad_mean > 0 else np.nan
        else:
            stat_row['FC_AD_vs_CG'] = np.nan
            stat_row['log2FC_AD_vs_CG'] = np.nan
    else:
        stat_row['FC_AD_vs_CG'] = np.nan
        stat_row['log2FC_AD_vs_CG'] = np.nan

    key_met_stats.append(stat_row)

key_met_df = pd.DataFrame(key_met_stats)
key_met_df.to_csv(r"C:\Users\PC\Documents\非靶向代谢组学分析\Data_Verification\key_metabolites_stats.csv", index=False, encoding='utf-8-sig')
print(f"\nKey metabolite stats saved ({len(key_met_stats)} metabolites)")

# ========== RF Biomarker Details ==========
print("\n--- RF Biomarker Details Extraction ---")

rf_biomarkers = [
    'metab_1202', 'metab_3622', 'metab_5514', 'metab_9224', 'metab_9920',
    'metab_9982', 'metab_10695', 'metab_12784', 'metab_13722', 'metab_19716'
]

rf_details = []

for metab_id in rf_biomarkers:
    matches = meta_df[meta_df['metab_id'] == metab_id]
    if len(matches) == 0:
        print(f"  WARNING: {metab_id} not found!")
        continue

    met_row = matches.iloc[0]
    csv_id = met_row['ID']
    met_name = met_row['Metabolite']
    kegg_id = met_row['KEGG Compound ID']
    mz = met_row['M/Z']
    rt = met_row['Retention time']
    mode = met_row['Mode']
    level = met_row['level']
    formula = met_row['Formula']
    library_id = met_row['Library ID']

    print(f"  Found: {metab_id} | {met_name} | ID={csv_id} | KEGG={kegg_id}")

    detail_row = {
        'metab_id': metab_id,
        'CSV_ID': csv_id,
        'Metabolite': met_name,
        'KEGG_ID': kegg_id,
        'Library_ID': library_id,
        'Formula': formula,
        'MZ': mz,
        'RT': rt,
        'Mode': mode,
        'Level': level
    }

    # Group means ± SD
    for grp in group_order:
        grp_cols = [c for c in exp_sample_cols if sample_groups[c] == grp]
        if len(grp_cols) > 0:
            vals = pd.to_numeric(met_row[grp_cols], errors='coerce').dropna()
            detail_row[f'{grp}_mean'] = vals.mean()
            detail_row[f'{grp}_sd'] = vals.std()
        else:
            detail_row[f'{grp}_mean'] = np.nan
            detail_row[f'{grp}_sd'] = np.nan

    # ANOVA
    groups_data = []
    for grp in group_order:
        grp_cols = [c for c in exp_sample_cols if sample_groups[c] == grp]
        if len(grp_cols) > 0:
            vals = pd.to_numeric(met_row[grp_cols], errors='coerce').dropna()
            if len(vals) > 0:
                groups_data.append(vals.values)

    if len(groups_data) >= 2:
        try:
            f_stat, p_val = stats.f_oneway(*groups_data)
            detail_row['ANOVA_P'] = p_val
        except:
            detail_row['ANOVA_P'] = np.nan
    else:
        detail_row['ANOVA_P'] = np.nan

    rf_details.append(detail_row)

rf_df = pd.DataFrame(rf_details)
rf_df.to_csv(r"C:\Users\PC\Documents\非靶向代谢组学分析\Data_Verification\rf_biomarker_details.csv", index=False, encoding='utf-8-sig')
print(f"\nRF biomarker details saved ({len(rf_details)} biomarkers)")

# ========== Summary ==========
print("\n" + "=" * 60)
print("DATA VERIFICATION SUMMARY")
print("=" * 60)
print(f"Total subjects: {len(crf_df)}")
print(f"Groups: {dict(group_counts)}")
print(f"Key metabolites found: {sum(1 for r in key_met_stats if r['Status'] == 'FOUND')}/{len(key_metabolites)}")
print(f"RF biomarkers found: {len(rf_details)}/{len(rf_biomarkers)}")
print(f"\nOutput files:")
print(f"  Tables/Table_01_Demographics.csv")
print(f"  Data_Verification/key_metabolites_stats.csv")
print(f"  Data_Verification/rf_biomarker_details.csv")
