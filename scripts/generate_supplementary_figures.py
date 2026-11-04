"""
生成三项补充分析的可视化图表
1. 置换检验分布图
2. 敏感性分析对比图
3. 反事实对比图
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8

BASE_DIR = r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析'
RESULTS_DIR = f'{BASE_DIR}\\Analysis_Results'
OUTPUT_DIR = f'{BASE_DIR}\\Supplementary_Figures'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 颜色方案
COLORS = {
    'uncorrected': '#E74C3C',
    'linear': '#3498DB',
    'combat': '#2ECC71',
    'binning': '#F39C12',
    'corrected': '#3498DB',
    'highlight': '#E74C3C',
    'neutral': '#95A5A6'
}

# ========== 图1：置换检验分布图 ==========
print("生成图1：置换检验分布图...")

with open(f'{RESULTS_DIR}\\analysis1_results.json', 'r') as f:
    a1 = json.load(f)

perm_accs = np.load(f'{RESULTS_DIR}\\analysis1_perm_accuracies.npy')
boot_accs = np.load(f'{RESULTS_DIR}\\analysis1_boot_accuracies.npy')

obs_acc = a1['repeated10_accuracy'] * 100
perm_mean = a1['permutation_mean'] * 100
perm_std = a1['permutation_std'] * 100
p_value = a1['permutation_p_value']
ci_lower = a1['bootstrap_ci_lower'] * 100
ci_upper = a1['bootstrap_ci_upper'] * 100

fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=600)

# 置换零分布直方图
n, bins, patches = ax.hist(perm_accs * 100, bins=30, color='#BDC3C7', 
                             edgecolor='white', linewidth=0.5, alpha=0.8,
                             label=f'Permutation null (n={len(perm_accs)})')

# 观测准确率竖线
ax.axvline(x=obs_acc, color=COLORS['highlight'], linewidth=2, linestyle='-',
           label=f'Observed accuracy = {obs_acc:.1f}%')

# Bootstrap CI区间
ax.axvspan(ci_lower, ci_upper, alpha=0.2, color=COLORS['corrected'],
           label=f'Bootstrap 95% CI [{ci_lower:.1f}%, {ci_upper:.1f}%]')

# 随机水平
ax.axvline(x=25, color='black', linewidth=1, linestyle='--', alpha=0.5)
ax.text(25.5, ax.get_ylim()[1] * 0.9, 'Chance\n(25%)', fontsize=8, 
        color='black', alpha=0.7, va='top')

# p值标注
ax.text(0.98, 0.95, f'p = {p_value:.4f}', transform=ax.transAxes,
        fontsize=11, fontweight='bold', ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#BDC3C7'))

ax.set_xlabel('Classification Accuracy (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Permutation Test for Corrected RF Classification', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.set_xlim(10, 65)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}\\Fig_S1_Permutation_Test.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}\\Fig_S1_Permutation_Test.pdf', bbox_inches='tight')
plt.close()
print(f"  保存到: {OUTPUT_DIR}\\Fig_S1_Permutation_Test.png")

# ========== 图2：敏感性分析对比图 ==========
print("生成图2：敏感性分析对比图...")

with open(f'{RESULTS_DIR}\\analysis2_results.json', 'r') as f:
    a2 = json.load(f)

methods = ['Uncorrected', 'Linear\nRegression', 'ComBat', 'Age\nBinning']
method_keys = ['Uncorrected', 'Method1_LinearRegression', 'Method2_ComBat', 'Method3_AgeBinning']
method_colors = [COLORS['uncorrected'], COLORS['linear'], COLORS['combat'], COLORS['binning']]

accuracies = [a2['methods'][k]['accuracy'] * 100 for k in method_keys]
sens_ad = [a2['methods'][k]['sensitivities']['AD'] * 100 for k in method_keys]
sens_cg = [a2['methods'][k]['sensitivities']['CG'] * 100 for k in method_keys]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=600)

# 左图：准确率对比
ax1 = axes[0]
bars = ax1.bar(methods, accuracies, color=method_colors, edgecolor='white', linewidth=0.8, width=0.6)
ax1.axhline(y=25, color='black', linewidth=1, linestyle='--', alpha=0.5)
ax1.text(0.02, 26, 'Chance (25%)', fontsize=8, color='black', alpha=0.7)

for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title('(a) Overall Accuracy', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 75)
ax1.tick_params(axis='x', labelsize=9)

# 右图：AD和CG灵敏度对比
ax2 = axes[1]
x = np.arange(len(methods))
width = 0.35

bars1 = ax2.bar(x - width/2, sens_ad, width, label='AD sensitivity', 
                 color=COLORS['corrected'], edgecolor='white', linewidth=0.8, alpha=0.8)
bars2 = ax2.bar(x + width/2, sens_cg, width, label='CG sensitivity',
                 color=COLORS['uncorrected'], edgecolor='white', linewidth=0.8, alpha=0.8)

for bar, val in zip(bars1, sens_ad):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, sens_cg):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=9)
ax2.set_ylabel('Sensitivity (%)', fontsize=11)
ax2.set_title('(b) Per-Class Sensitivity', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}\\Fig_S2_Sensitivity_Analysis.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}\\Fig_S2_Sensitivity_Analysis.pdf', bbox_inches='tight')
plt.close()
print(f"  保存到: {OUTPUT_DIR}\\Fig_S2_Sensitivity_Analysis.png")

# ========== 图3：反事实对比图 ==========
print("生成图3：反事实对比图...")

with open(f'{RESULTS_DIR}\\analysis3_results.json', 'r') as f:
    a3 = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=600)

# (a) 准确率对比
ax1 = axes[0]
conditions = ['Uncorrected\n(B(i), no age adj.)', 'Corrected\n(B(i), age adj.)']
acc_vals = [a3['uncorrected']['accuracy'] * 100, a3['corrected']['accuracy'] * 100]
bars = ax1.bar(conditions, acc_vals, color=[COLORS['uncorrected'], COLORS['corrected']],
                edgecolor='white', linewidth=0.8, width=0.5)
ax1.axhline(y=25, color='black', linewidth=1, linestyle='--', alpha=0.5)

for bar, val in zip(bars, acc_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 差异标注
diff = acc_vals[0] - acc_vals[1]
ax1.annotate(f'-{diff:.1f} pp', xy=(1, acc_vals[1] + 5), fontsize=9,
             ha='center', color=COLORS['highlight'], fontweight='bold')

ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title('(a) Classification Accuracy', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 75)

# (b) Top-50中B(ii)级比例
ax2 = axes[1]
bii_vals = [a3['uncorrected']['top50_analysis']['level_percentages']['B(ii)'],
             a3['corrected']['top50_analysis']['level_percentages']['B(ii)']]
bars = ax2.bar(conditions, bii_vals, color=[COLORS['uncorrected'], COLORS['corrected']],
                edgecolor='white', linewidth=0.8, width=0.5)

for bar, val in zip(bars, bii_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('B(ii)-level Metabolites (%)', fontsize=11)
ax2.set_title('(b) Xenobiotic Contamination\nin Top-50 Biomarkers', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 45)

# (c) NMF亚型间年龄差
ax3 = axes[2]
age_diff_uncorr = abs(a3['uncorrected']['nmf_analysis']['age_mean_diff'])
age_diff_corr = abs(a3['corrected']['nmf_analysis']['age_mean_diff'])
p_uncorr = a3['uncorrected']['nmf_analysis']['age_p_value']
p_corr = a3['corrected']['nmf_analysis']['age_p_value']

age_vals = [age_diff_uncorr, age_diff_corr]
bars = ax3.bar(conditions, age_vals, color=[COLORS['uncorrected'], COLORS['corrected']],
                edgecolor='white', linewidth=0.8, width=0.5)

for bar, val, p in zip(bars, age_vals, [p_uncorr, p_corr]):
    sig = '*' if p < 0.05 else 'ns'
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f} yr\n({sig})', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax3.set_ylabel('Age Difference Between\nNMF Subtypes (years)', fontsize=11)
ax3.set_title('(c) Age Confounding in\nNMF Subtyping', fontsize=11, fontweight='bold')
ax3.set_ylim(0, 7.5)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}\\Fig_S3_Counterfactual_Comparison.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}\\Fig_S3_Counterfactual_Comparison.pdf', bbox_inches='tight')
plt.close()
print(f"  保存到: {OUTPUT_DIR}\\Fig_S3_Counterfactual_Comparison.png")

print("\n" + "=" * 60)
print("所有图表生成完成！")
print(f"输出目录: {OUTPUT_DIR}")
print("=" * 60)
