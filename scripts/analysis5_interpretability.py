"""
分析5：可解释性分析（基于scikit-learn，替代SHAP）
1. Permutation importance（与MDG对比）
2. Partial dependence plots（关键代谢物与AD概率的关系）
3. 个体预测解释（典型样本的特征贡献）
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析')
from analysis_utils import load_expression_matrix, load_clinical_data, filter_by_level, log_transform, age_residualization, get_sample_groups, GROUP_ORDER

OUTPUT_DIR = r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析\Analysis_Results'
FIG_DIR = r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析\Supplementary_Figures'
RANDOM_STATE = 42
N_ESTIMATORS = 1000

print("=" * 70)
print("分析5：可解释性分析（Permutation Importance + Partial Dependence）")
print("=" * 70)

# ========== 1. 加载数据 ==========
print("\n[1/5] 加载数据...")
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

# 划分训练集和测试集（用于permutation importance）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)
print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ========== 2. 训练RF ==========
print("\n[2/5] 训练Random Forest...")
rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features='sqrt',
                              random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))
test_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"  训练集准确率: {train_acc:.1%}")
print(f"  测试集准确率: {test_acc:.1%}")

# MDG重要性
mdg_importance = rf.feature_importances_

# ========== 3. Permutation Importance（只对Top-100特征计算，加速） ==========
print("\n[3/5] 计算Permutation Importance（Top-100特征）...")

# 先按MDG选出Top-100特征
top100_idx = np.argsort(mdg_importance)[-100:][::-1]
X_test_top100 = X_test[:, top100_idx]
X_train_top100 = X_train[:, top100_idx]

# 用Top-100特征重新训练RF（用于permutation importance）
rf_top100 = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                     random_state=RANDOM_STATE, n_jobs=-1)
rf_top100.fit(X_train_top100, y_train)

perm_result = permutation_importance(
    rf_top100, X_test_top100, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
)
perm_mean_top100 = perm_result.importances_mean
perm_std_top100 = perm_result.importances_std

# 映射回原始特征索引
perm_mean = np.zeros(X.shape[1])
perm_std = np.zeros(X.shape[1])
for i, orig_idx in enumerate(top100_idx):
    perm_mean[orig_idx] = perm_mean_top100[i]
    perm_std[orig_idx] = perm_std_top100[i]

# Top-20代谢物（按permutation importance）
top20_idx = np.argsort(perm_mean)[-20:][::-1]
top20_names = [metabolite_names[i] for i in top20_idx]
top20_perm = perm_mean[top20_idx]
top20_mdg = mdg_importance[top20_idx]

print(f"  Top-10 by Permutation Importance:")
for i in range(10):
    print(f"    {i+1}. {top20_names[i][:40]:40s}: perm={top20_perm[i]:.4f}, mdg={top20_mdg[i]:.4f}")

# 计算MDG和Permutation的相关性
from scipy.stats import spearmanr
corr, p_val = spearmanr(mdg_importance, perm_mean)
print(f"\n  MDG与Permutation Importance的Spearman相关: rho={corr:.3f}, p={p_val:.4f}")

# ========== 4. Partial Dependence Plots ==========
print("\n[4/5] 计算Partial Dependence...")

# 对Top-5代谢物，计算对AD类别的partial dependence
ad_class_idx = list(rf.classes_).index('AD')
top5_idx = top20_idx[:5]
top5_names = [metabolite_names[i] for i in top5_idx]

pdp_results = {}
for i, idx in enumerate(top5_idx):
    print(f"  计算 {top5_names[i][:30]} 的partial dependence...")
    try:
        pdp = partial_dependence(
            rf, X_train, features=[idx], target=ad_class_idx,
            grid_resolution=20, method='brute'
        )
        pdp_results[top5_names[i]] = {
            'grid_values': pdp['grid_values'][0].tolist(),
            'average': pdp['average'][0].tolist(),
            'feature_idx': int(idx),
        }
    except Exception as e:
        print(f"    错误: {e}")

# ========== 5. 个体预测解释 ==========
print("\n[5/5] 个体预测解释...")

# 选择一个典型AD样本和一个典型CG样本
ad_mask = y_test == 'AD'
cg_mask = y_test == 'CG'

if ad_mask.sum() > 0:
    ad_sample_idx = np.where(ad_mask)[0][0]
    ad_proba = rf.predict_proba(X_test[ad_sample_idx:ad_sample_idx+1])[0]
    print(f"  典型AD样本预测概率: {dict(zip(rf.classes_, ad_proba))}")
    
    # 计算该样本的特征贡献（基于树路径）
    # 简化方法：计算该样本特征值与均值的差异 × feature importance
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    sample_contribution = (X_test[ad_sample_idx] - X_mean) * mdg_importance
    top_contributors_ad = np.argsort(np.abs(sample_contribution))[-10:][::-1]
    
    print(f"  AD样本Top-10贡献代谢物:")
    for i in top_contributors_ad[:5]:
        direction = "升高" if sample_contribution[i] > 0 else "降低"
        print(f"    {metabolite_names[i][:40]:40s}: {direction} (贡献={sample_contribution[i]:.4f})")

if cg_mask.sum() > 0:
    cg_sample_idx = np.where(cg_mask)[0][0]
    cg_proba = rf.predict_proba(X_test[cg_sample_idx:cg_sample_idx+1])[0]
    print(f"  典型CG样本预测概率: {dict(zip(rf.classes_, cg_proba))}")

# ========== 生成图表 ==========
print("\n生成图表...")

# 图1: Permutation Importance vs MDG
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

# 左图: Permutation Importance Top-20
ax1 = axes[0]
y_pos = np.arange(20)
ax1.barh(y_pos, top20_perm[::-1], xerr=perm_std[top20_idx][::-1],
         color='#3498DB', alpha=0.8, edgecolor='white')
ax1.set_yticks(y_pos)
ax1.set_yticklabels([n[:25] for n in top20_names[::-1]], fontsize=7)
ax1.set_xlabel('Permutation Importance (accuracy decrease)', fontsize=10)
ax1.set_title('(a) Permutation Importance (Top 20)', fontsize=11, fontweight='bold')

# 右图: MDG vs Permutation散点图
ax2 = axes[1]
ax2.scatter(mdg_importance, perm_mean, alpha=0.3, s=10, color='#2C3E50')
# 标注Top-5
for i in top20_idx[:5]:
    ax2.annotate(metabolite_names[i][:15], (mdg_importance[i], perm_mean[i]),
                 fontsize=6, alpha=0.8)
ax2.set_xlabel('MDG (Mean Decrease in Gini)', fontsize=10)
ax2.set_ylabel('Permutation Importance', fontsize=10)
ax2.set_title(f'(b) MDG vs Permutation (rho={corr:.3f})', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}\\Fig_S4_Interpretability_Importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  保存: Fig_S4_Interpretability_Importance.png")

# 图2: Partial Dependence Plots
if pdp_results:
    n_pdp = min(5, len(pdp_results))
    fig, axes = plt.subplots(1, n_pdp, figsize=(4*n_pdp, 3.5), dpi=300)
    if n_pdp == 1:
        axes = [axes]
    
    for i, (name, data) in enumerate(list(pdp_results.items())[:n_pdp]):
        ax = axes[i]
        ax.plot(data['grid_values'], data['average'], color='#E74C3C', linewidth=2)
        ax.fill_between(data['grid_values'], data['average'], alpha=0.2, color='#E74C3C')
        ax.set_xlabel(name[:20], fontsize=8)
        ax.set_ylabel('P(AD)' if i == 0 else '', fontsize=9)
        ax.set_title(f'({chr(97+i)}) {name[:25]}', fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}\\Fig_S5_Partial_Dependence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: Fig_S5_Partial_Dependence.png")

# ========== 保存结果 ==========
print("\n保存结果...")

output = {
    'mdg_vs_permutation_correlation': float(corr),
    'mdg_vs_permutation_p': float(p_val),
    'top20_permutation_importance': [
        {
            'rank': i+1,
            'metabolite': top20_names[i],
            'permutation_mean': float(top20_perm[i]),
            'permutation_std': float(perm_std[top20_idx[i]]),
            'mdg': float(top20_mdg[i]),
        } for i in range(20)
    ],
    'partial_dependence': {k: v for k, v in pdp_results.items()},
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
}

with open(f'{OUTPUT_DIR}\\analysis5_interpretability.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"  结果已保存到: {OUTPUT_DIR}")
print("\n" + "=" * 70)
print("关键发现")
print("=" * 70)
print(f"1. MDG与Permutation Importance相关性: rho={corr:.3f} (p={p_val:.4f})")
print(f"2. Top-5 by Permutation Importance:")
for i in range(5):
    print(f"   {i+1}. {top20_names[i][:40]}")
print(f"3. 训练集准确率: {train_acc:.1%}, 测试集准确率: {test_acc:.1%}")
