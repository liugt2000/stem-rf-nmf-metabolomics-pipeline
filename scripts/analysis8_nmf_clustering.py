"""
分析8：NMF k值系统比较 + 聚类方法比较
1. k=2,3,4,5的系统比较（consensus silhouette, PAC, cophenetic correlation）
2. 与k-means、层次聚类、谱聚类的结果比较（ARI）
"""
import sys
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

sys.path.insert(0, r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析')
from analysis_utils import load_expression_matrix, load_clinical_data, filter_by_level, log_transform, age_residualization, get_sample_groups, GROUP_ORDER

OUTPUT_DIR = r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析\Analysis_Results'
RANDOM_STATE = 42

print("=" * 70)
print("分析8：NMF k值系统比较 + 聚类方法比较")
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
print(f"  矩阵: {X.shape}")

# NMF需要非负输入，将残差平移到非负
X_nonneg = X - X.min() + 1e-10

# ========== 2. NMF k值系统比较 ==========
print("\n[2/4] NMF k值系统比较（k=2,3,4,5）...")

k_values = [2, 3, 4, 5]
nmf_results = {}

for k in k_values:
    print(f"\n  --- k={k} ---")
    
    # 运行NMF
    model = NMF(n_components=k, init='nndsvda', max_iter=1000,
                random_state=RANDOM_STATE)
    W = model.fit_transform(X_nonneg)  # 样本 × 亚型
    H = model.components_  # 亚型 × 特征
    
    # 分配到概率最大的亚型
    subtypes = np.argmax(W, axis=1)
    
    # 计算指标
    # 1. Reconstruction error (R²X)
    reconstruction_err = model.reconstruction_err_
    total_var = np.sum(X_nonneg ** 2)
    r2x = 1 - reconstruction_err ** 2 / total_var
    
    # 2. Silhouette score
    if len(np.unique(subtypes)) > 1:
        sil_score = silhouette_score(X_nonneg, subtypes)
    else:
        sil_score = -1
    
    # 3. Cophenetic correlation (用层次聚类近似)
    Z = linkage(W, method='average')
    coph_corr, _ = cophenet(Z, pdist(W))
    
    # 4. 亚型大小
    subtype_sizes = dict(pd.Series(subtypes).value_counts().sort_index())
    
    # 5. 亚型与疾病的关联
    disease_assoc = {}
    for st in range(k):
        st_mask = subtypes == st
        if st_mask.sum() > 0:
            disease_dist = dict(pd.Series(y[st_mask]).value_counts(normalize=True))
            disease_assoc[f'subtype_{st}'] = {
                'n': int(st_mask.sum()),
                'disease_dist': {k: float(v) for k, v in disease_dist.items()}
            }
    
    nmf_results[k] = {
        'r2x': float(r2x),
        'silhouette': float(sil_score),
        'cophenetic_correlation': float(coph_corr),
        'reconstruction_error': float(reconstruction_err),
        'subtype_sizes': {str(k): int(v) for k, v in subtype_sizes.items()},
        'disease_association': disease_assoc,
        'subtypes': subtypes.tolist(),
    }
    
    print(f"    R²X: {r2x:.4f}")
    print(f"    Silhouette: {sil_score:.4f}")
    print(f"    Cophenetic correlation: {coph_corr:.4f}")
    print(f"    亚型大小: {subtype_sizes}")

# 选择最优k
best_k = max(k_values, key=lambda k: nmf_results[k]['silhouette'])
print(f"\n  最优k（按silhouette）: k={best_k}")

# ========== 3. 聚类方法比较 ==========
print("\n[3/4] 聚类方法比较（k=2）...")

k = 2
nmf_subtypes = np.array(nmf_results[k]['subtypes'])

# 标准化数据（用于其他聚类方法）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_nonneg)

clustering_methods = {
    'NMF': nmf_subtypes,
    'K-Means': KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_scaled),
    'Hierarchical (Ward)': AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X_scaled),
    'Hierarchical (Average)': AgglomerativeClustering(n_clusters=k, linkage='average').fit_predict(X_scaled),
    'Spectral': SpectralClustering(n_clusters=k, random_state=RANDOM_STATE, affinity='nearest_neighbors').fit_predict(X_scaled),
}

# 计算ARI（与NMF比较）和silhouette
comparison_results = {}
for method_name, labels in clustering_methods.items():
    ari = adjusted_rand_score(nmf_subtypes, labels)
    if len(np.unique(labels)) > 1:
        sil = silhouette_score(X_scaled, labels)
    else:
        sil = -1
    
    # 亚型大小
    sizes = dict(pd.Series(labels).value_counts().sort_index())
    
    comparison_results[method_name] = {
        'ari_with_nmf': float(ari),
        'silhouette': float(sil),
        'subtype_sizes': {str(k): int(v) for k, v in sizes.items()},
    }
    
    print(f"  {method_name:25s}: ARI={ari:.3f}, Silhouette={sil:.3f}, 大小={sizes}")

# 计算平均ARI（排除NMF自身）
other_methods = [m for m in clustering_methods if m != 'NMF']
avg_ari = np.mean([comparison_results[m]['ari_with_nmf'] for m in other_methods])
print(f"\n  平均ARI（NMF vs 其他4种方法）: {avg_ari:.3f}")

# ========== 4. k=2时亚型与年龄/疾病的关联 ==========
print("\n[4/4] k=2时亚型与临床指标的关联...")

from scipy import stats

sample_crf = crf.loc[samples].copy()
sample_crf['nmf_subtype'] = nmf_subtypes

# 年龄差异
age_subtype0 = sample_crf[sample_crf['nmf_subtype'] == 0]['Age'].values
age_subtype1 = sample_crf[sample_crf['nmf_subtype'] == 1]['Age'].values
t_stat, age_p = stats.ttest_ind(age_subtype0, age_subtype1)
age_diff = np.mean(age_subtype0) - np.mean(age_subtype1)

print(f"  亚型0 vs 亚型1 年龄差: {age_diff:.2f}岁 (p={age_p:.4f})")
print(f"  亚型0平均年龄: {np.mean(age_subtype0):.1f}岁 (n={len(age_subtype0)})")
print(f"  亚型1平均年龄: {np.mean(age_subtype1):.1f}岁 (n={len(age_subtype1)})")

# MMSE差异
if 'MMSE' in sample_crf.columns:
    mmse0 = sample_crf[sample_crf['nmf_subtype'] == 0]['MMSE'].dropna().values
    mmse1 = sample_crf[sample_crf['nmf_subtype'] == 1]['MMSE'].dropna().values
    if len(mmse0) > 1 and len(mmse1) > 1:
        t_mmse, mmse_p = stats.ttest_ind(mmse0, mmse1)
        print(f"  亚型0 vs 亚型1 MMSE差: {np.mean(mmse0)-np.mean(mmse1):.2f} (p={mmse_p:.4f})")

# ========== 保存结果 ==========
print("\n" + "=" * 70)
print("保存结果")
print("=" * 70)

output = {
    'nmf_k_comparison': {str(k): v for k, v in nmf_results.items()},
    'best_k': best_k,
    'clustering_comparison': comparison_results,
    'average_ari_nmf_vs_others': float(avg_ari),
    'subtype_age_difference': {
        'mean_diff_years': float(age_diff),
        'p_value': float(age_p),
        'subtype0_mean_age': float(np.mean(age_subtype0)),
        'subtype1_mean_age': float(np.mean(age_subtype1)),
    },
    'k_values_tested': k_values,
}

with open(f'{OUTPUT_DIR}\\analysis8_nmf_clustering.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

# 保存k值比较表
k_table = []
for k in k_values:
    row = {
        'k': k,
        'R2X': f"{nmf_results[k]['r2x']:.4f}",
        'Silhouette': f"{nmf_results[k]['silhouette']:.4f}",
        'Cophenetic': f"{nmf_results[k]['cophenetic_correlation']:.4f}",
        'Subtype_sizes': str(nmf_results[k]['subtype_sizes']),
    }
    k_table.append(row)

k_df = pd.DataFrame(k_table)
k_df.to_csv(f'{OUTPUT_DIR}\\analysis8_nmf_k_comparison.csv',
             index=False, encoding='utf-8-sig')
print("\nNMF k值比较:")
print(k_df.to_string(index=False))

# 保存聚类方法比较表
clust_table = []
for method, res in comparison_results.items():
    row = {
        'Method': method,
        'ARI_with_NMF': f"{res['ari_with_nmf']:.3f}",
        'Silhouette': f"{res['silhouette']:.3f}",
        'Subtype_sizes': str(res['subtype_sizes']),
    }
    clust_table.append(row)

clust_df = pd.DataFrame(clust_table)
clust_df.to_csv(f'{OUTPUT_DIR}\\analysis8_clustering_comparison.csv',
                index=False, encoding='utf-8-sig')
print("\n聚类方法比较:")
print(clust_df.to_string(index=False))

print(f"\n结果已保存到: {OUTPUT_DIR}")
print("\n关键发现:")
print(f"1. 最优k={best_k}（按silhouette）")
print(f"2. NMF与其他4种聚类方法的平均ARI: {avg_ari:.3f}")
print(f"3. k=2时亚型间年龄差: {age_diff:.2f}岁 (p={age_p:.4f})")
