"""
分析9：代谢物-临床指标关联分析
Top-50代谢物与MMSE、HAMD、HAMA、病程、年龄的Spearman相关
"""
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析')
from analysis_utils import load_expression_matrix, load_clinical_data, filter_by_level, log_transform, age_residualization, get_sample_groups, GROUP_ORDER

OUTPUT_DIR = r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析\Analysis_Results'
RANDOM_STATE = 42

print("=" * 70)
print("分析9：代谢物-临床指标关联分析")
print("=" * 70)

# ========== 1. 加载数据 ==========
print("\n[1/4] 加载数据...")
annot, expr = load_expression_matrix()
crf = load_clinical_data()

annot_bi, expr_bi = filter_by_level(annot, expr, 'B(i)')
expr_log = log_transform(expr_bi)
expr_resid = age_residualization(expr_log, crf)

samples, groups = get_sample_groups(expr_resid, crf)
X = expr_resid[samples].values.T
y = np.array(groups)
metabolite_ids = expr_bi.index.tolist()
metabolite_names = annot_bi['Metabolite'].values

print(f"  矩阵: {X.shape}")
print(f"  临床指标: {list(crf.columns)}")

# ========== 2. 训练RF获取Top-50代谢物 ==========
print("\n[2/4] 训练RF获取Top-50代谢物...")

rf = RandomForestClassifier(n_estimators=1000, max_features='sqrt',
                              random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X, y)
importances = rf.feature_importances_
top50_idx = np.argsort(importances)[-50:][::-1]
top50_ids = [metabolite_ids[i] for i in top50_idx]
top50_names = [metabolite_names[i] for i in top50_idx]

print(f"  Top-50代谢物已获取")
print(f"  Top-5: {top50_names[:5]}")

# ========== 3. 计算与临床指标的相关性 ==========
print("\n[3/4] 计算与临床指标的相关性...")

# 临床指标
clinical_vars = ['MMSE', 'HAMD', 'HAMA', 'Age']
# 检查是否有病程字段
if 'Course' in crf.columns:
    clinical_vars.append('Course')
elif '病程' in crf.columns:
    clinical_vars.append('病程')

print(f"  临床指标: {clinical_vars}")

# 获取样本的临床数据
sample_clinical = crf.loc[samples].copy()

# 计算每个Top-50代谢物与每个临床指标的Spearman相关
correlation_results = []

for i, idx in enumerate(top50_idx):
    metab_values = X[:, idx]
    metab_name = top50_names[i]
    metab_id = top50_ids[i]
    
    result = {
        'rank': i + 1,
        'metabolite_id': metab_id,
        'metabolite_name': metab_name,
        'rf_importance': float(importances[idx]),
    }
    
    for var in clinical_vars:
        if var in sample_clinical.columns:
            clin_values = sample_clinical[var].values
            # 去除NaN
            valid = ~np.isnan(clin_values) & ~np.isnan(metab_values)
            if valid.sum() > 10:
                rho, p_val = spearmanr(metab_values[valid], clin_values[valid])
                result[f'{var}_rho'] = float(rho)
                result[f'{var}_p'] = float(p_val)
                result[f'{var}_n'] = int(valid.sum())
            else:
                result[f'{var}_rho'] = None
                result[f'{var}_p'] = None
    
    correlation_results.append(result)

# 转换为DataFrame
corr_df = pd.DataFrame(correlation_results)

# FDR校正（BH方法，手动实现）
def bh_fdr(p_values):
    """Benjamini-Hochberg FDR校正"""
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    fdr = np.zeros(n)
    for i in range(n):
        fdr[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
    # 确保单调
    for i in range(n-2, -1, -1):
        fdr[sorted_idx[i]] = min(fdr[sorted_idx[i]], fdr[sorted_idx[i+1]])
    return np.clip(fdr, 0, 1)

for var in clinical_vars:
    p_col = f'{var}_p'
    if p_col in corr_df.columns:
        valid_p = corr_df[p_col].dropna()
        if len(valid_p) > 0:
            fdr_p = bh_fdr(valid_p.values)
            corr_df.loc[valid_p.index, f'{var}_fdr'] = fdr_p

print(f"  相关性计算完成")
print(f"  结果: {corr_df.shape}")

# 显示Top相关
print("\n  与MMSE相关性最强的Top-5代谢物:")
mmse_col = 'MMSE_rho'
if mmse_col in corr_df.columns:
    top_mmse = corr_df.reindex(corr_df[mmse_col].abs().sort_values(ascending=False).index).head(5)
    for _, row in top_mmse.iterrows():
        print(f"    {row['metabolite_name'][:40]:40s}: rho={row[mmse_col]:.3f}, p={row['MMSE_p']:.4f}, FDR={row.get('MMSE_fdr', 'N/A')}")

# ========== 4. 疾病四阶段趋势分析 ==========
print("\n[4/4] 疾病四阶段趋势分析...")

# 对每个Top-10代谢物，计算在四阶段的均值
top10_idx = top50_idx[:10]
trend_results = []

for i, idx in enumerate(top10_idx):
    metab_values = X[:, idx]
    metab_name = top50_names[i]
    
    stage_means = {}
    for stage in GROUP_ORDER:
        stage_mask = y == stage
        stage_means[stage] = float(np.mean(metab_values[stage_mask]))
    
    # 计算趋势（从CG到AD的斜率）
    stage_order = GROUP_ORDER  # CG, SCD, MCI, AD
    x_vals = np.arange(len(stage_order))
    y_vals = [stage_means[s] for s in stage_order]
    slope = np.polyfit(x_vals, y_vals, 1)[0]
    
    trend_results.append({
        'rank': i + 1,
        'metabolite_name': metab_name,
        'CG_mean': stage_means['CG'],
        'SCD_mean': stage_means['SCD'],
        'MCI_mean': stage_means['MCI'],
        'AD_mean': stage_means['AD'],
        'trend_slope': float(slope),
    })

trend_df = pd.DataFrame(trend_results)
print("\n  Top-10代谢物四阶段趋势:")
print(trend_df[['rank', 'metabolite_name', 'CG_mean', 'SCD_mean', 'MCI_mean', 'AD_mean', 'trend_slope']].to_string(index=False))

# ========== 保存结果 ==========
print("\n" + "=" * 70)
print("保存结果")
print("=" * 70)

output = {
    'top50_correlations': correlation_results,
    'clinical_variables': clinical_vars,
    'top10_trends': trend_results,
    'n_samples': len(samples),
    'n_metabolites': X.shape[1],
}

with open(f'{OUTPUT_DIR}\\analysis9_clinical_correlation.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

corr_df.to_csv(f'{OUTPUT_DIR}\\analysis9_clinical_correlation.csv',
                index=False, encoding='utf-8-sig')
trend_df.to_csv(f'{OUTPUT_DIR}\\analysis9_stage_trends.csv',
                 index=False, encoding='utf-8-sig')

print(f"  结果已保存到: {OUTPUT_DIR}")
print("\n关键发现:")
# 统计显著相关的代谢物数量
for var in clinical_vars:
    fdr_col = f'{var}_fdr'
    if fdr_col in corr_df.columns:
        n_sig = (corr_df[fdr_col] < 0.05).sum()
        print(f"  与{var}显著相关 (FDR<0.05): {n_sig} 个代谢物")
