# -*- coding: utf-8 -*-
"""
Step 1 Fix: Clinical CRF data processing - generate proper Table 1
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========== 1. Clinical CRF Data Processing ==========
crf_path = r"C:\Users\PC\Documents\非靶向代谢组学分析\临床CRF表.xls"
crf_df = pd.read_excel(crf_path)

# Fix Group column - strip whitespace
crf_df['Group'] = crf_df['Group'].astype(str).str.strip()

group_order = ['CG', 'SCD', 'MCI', 'AD']

# Exclude ID columns from Table 1
id_cols = ['Sample ID', 'Sample Analysis Name']
analysis_cols = [c for c in crf_df.columns if c not in id_cols and c != 'Group']

print("Columns for Table 1:", analysis_cols)
print("Group distribution:", crf_df['Group'].value_counts().to_dict())

# ========== Generate Table 1 ==========
table1_rows = []

for col in analysis_cols:
    col_data = crf_df[col]

    # Try numeric
    numeric_data = pd.to_numeric(col_data, errors='coerce')
    non_null_numeric = numeric_data.notna().sum()
    non_null_total = col_data.notna().sum()

    if non_null_total > 0 and non_null_numeric / non_null_total > 0.7:
        # === Numeric variable ===
        row = {'Variable': col}
        groups_for_test = []
        for grp in group_order:
            vals = numeric_data[crf_df['Group'] == grp].dropna()
            if len(vals) > 0:
                row[grp] = f"{vals.mean():.1f} ± {vals.std():.1f}"
                groups_for_test.append(vals.values)
            else:
                row[grp] = "N/A"

        # Kruskal-Wallis test
        if len(groups_for_test) >= 2:
            try:
                h_stat, p_val = stats.kruskal(*groups_for_test)
                row['P-value'] = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
            except:
                row['P-value'] = "N/A"
        else:
            row['P-value'] = "N/A"
        table1_rows.append(row)

    else:
        # === Categorical variable ===
        categories = sorted(col_data.dropna().astype(str).unique())
        for cat in categories:
            row = {'Variable': f"  {cat}"}
            for grp in group_order:
                grp_vals = col_data[crf_df['Group'] == grp].dropna()
                cat_count = (grp_vals.astype(str) == cat).sum()
                grp_total = len(grp_vals)
                if grp_total > 0:
                    row[grp] = f"{cat_count} ({cat_count/grp_total*100:.1f}%)"
                else:
                    row[grp] = "N/A"

            # Chi-square test for this category
            try:
                contingency = []
                for grp in group_order:
                    grp_vals = col_data[crf_df['Group'] == grp].dropna()
                    cat_count = (grp_vals.astype(str) == cat).sum()
                    contingency.append([cat_count, len(grp_vals) - cat_count])
                chi2, p_val, _, _ = stats.chi2_contingency(contingency)
                row['P-value'] = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
            except:
                row['P-value'] = "N/A"
            table1_rows.append(row)

        # Add header row for the variable
        header_idx = len(table1_rows) - len(categories)
        table1_rows.insert(header_idx, {'Variable': col, 'CG': '', 'SCD': '', 'MCI': '', 'AD': '', 'P-value': ''})

# Build DataFrame
table1_df = pd.DataFrame(table1_rows)
table1_df = table1_df[['Variable', 'CG', 'SCD', 'MCI', 'AD', 'P-value']]

# Add N row at top
n_row = pd.DataFrame([{
    'Variable': 'N',
    'CG': str(len(crf_df[crf_df['Group'] == 'CG'])),
    'SCD': str(len(crf_df[crf_df['Group'] == 'SCD'])),
    'MCI': str(len(crf_df[crf_df['Group'] == 'MCI'])),
    'AD': str(len(crf_df[crf_df['Group'] == 'AD'])),
    'P-value': ''
}])
table1_df = pd.concat([n_row, table1_df], ignore_index=True)

print("\n" + "=" * 80)
print("TABLE 1: Demographic and Clinical Characteristics")
print("=" * 80)
print(table1_df.to_string(index=False))

# Save
table1_df.to_csv(r"C:\Users\PC\Documents\非靶向代谢组学分析\Tables\Table_01_Demographics.csv", index=False, encoding='utf-8-sig')
print("\nTable 1 saved.")

# ========== Also generate Table 2: Differential Metabolites Summary ==========
print("\n" + "=" * 80)
print("TABLE 2: Differential Metabolites Summary (from PDF report)")
print("=" * 80)

diff_data = [
    {'Comparison': 'AD vs CG', 'Up': 226, 'Down': 631, 'Total': 857},
    {'Comparison': 'AD vs MCI', 'Up': 188, 'Down': 771, 'Total': 959},
    {'Comparison': 'AD vs SCD', 'Up': 215, 'Down': 849, 'Total': 1064},
    {'Comparison': 'MCI vs CG', 'Up': 337, 'Down': 264, 'Total': 601},
    {'Comparison': 'MCI vs SCD', 'Up': 175, 'Down': 99, 'Total': 274},
    {'Comparison': 'SCD vs CG', 'Up': 291, 'Down': 280, 'Total': 571},
]
table2_df = pd.DataFrame(diff_data)
print(table2_df.to_string(index=False))
table2_df.to_csv(r"C:\Users\PC\Documents\非靶向代谢组学分析\Tables\Table_02_Differential_Metabolites.csv", index=False, encoding='utf-8-sig')
print("\nTable 2 saved.")

# ========== Print key metabolite stats summary ==========
print("\n" + "=" * 80)
print("Key Metabolite Verification Summary")
print("=" * 80)

key_stats = pd.read_csv(r"C:\Users\PC\Documents\非靶向代谢组学分析\Data_Verification\key_metabolites_stats.csv")
for _, row in key_stats.iterrows():
    if row['Status'] == 'FOUND':
        print(f"\n{row['Metabolite']} (KEGG: {row['KEGG_ID']}, {row['Mode']}, {row['Level']})")
        for grp in group_order:
            mean_col = f'{grp}_mean'
            sd_col = f'{grp}_sd'
            if mean_col in row and pd.notna(row[mean_col]):
                print(f"  {grp}: {row[mean_col]:.4f} ± {row[sd_col]:.4f} (n={int(row[f'{grp}_n'])})")
        if 'ANOVA_P' in row and pd.notna(row['ANOVA_P']):
            p = row['ANOVA_P']
            print(f"  ANOVA P = {p:.4f}" if p >= 0.0001 else "  ANOVA P < 0.0001")
        if 'log2FC_AD_vs_CG' in row and pd.notna(row['log2FC_AD_vs_CG']):
            print(f"  log2FC (AD/CG) = {row['log2FC_AD_vs_CG']:.4f}")
