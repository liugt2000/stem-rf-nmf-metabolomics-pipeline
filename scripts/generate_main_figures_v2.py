# -*- coding: utf-8 -*-
"""
主图重绘 v2（全部数字来自 analysis JSON + verify_main_figures_data.py 重跑结果）
覆盖输出:
  Metabolites_Submission_Figures\Figure1-6.png + Graphical_Abstract.png (600 DPI)
  非靶向代谢组学分析\Supplementary_Figures\Fig_S6_Decomposition.png (docx 嵌入用)
运行: D:\ANAconda\envs\cptac_env\python.exe generate_main_figures_v2.py
依赖: main_figures_recompute.json (verify_main_figures_data.py 产物)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

BASE = r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version'
ANA = os.path.join(BASE, '非靶向代谢组学分析')
R = os.path.join(ANA, 'Analysis_Results')
FIGDIR = os.path.join(BASE, 'Metabolites_Submission_Figures')
SUP = os.path.join(ANA, 'Supplementary_Figures')
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(SUP, exist_ok=True)

sys.path.insert(0, ANA)
from analysis_utils import (load_expression_matrix, load_clinical_data, filter_by_level,
                            log_transform, age_residualization, get_sample_groups, GROUP_ORDER)

a1 = json.load(open(os.path.join(R, 'analysis1_results.json'), encoding='utf-8'))
a8 = json.load(open(os.path.join(R, 'analysis8_nmf_clustering.json'), encoding='utf-8'))
recomp = json.load(open(os.path.join(R, 'main_figures_recompute.json'), encoding='utf-8'))

COLORS = {'CG': '#4C72B0', 'SCD': '#DD8452', 'MCI': '#55A868', 'AD': '#C44E52'}
LABELS = {'CG': 'HC', 'SCD': 'SCD', 'MCI': 'MCI', 'AD': 'AD'}  # 论文统一对照组名为 HC
SUB_COLORS = ['#4C72B0', '#C44E52']  # Subtype 0 / 1

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10})


def save(fig, name, dpi=600):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('saved', path)


# ============================================================ Figure 1
def fig1():
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    ax.set_xlim(0, 1350); ax.set_ylim(0, 600); ax.axis('off')

    def box(x, y, w, h, title, body, fc='#EAF1F8', ec='#2F5597'):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=6', fc=fc, ec=ec, lw=1.6)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h - 22, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color='#1F3864')
        ax.text(x + w / 2, y + 30, body, ha='center', va='center', fontsize=9.2, color='#1F3864')

    def arrow(x1, y, x2):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle='-|>',
                                     mutation_scale=22, lw=2.0, color='#2F5597'))

    # 顶部信息条
    ax.text(675, 570, '84 fecal samples:  HC (n=10) | SCD (n=22) | MCI (n=23) | AD (n=29)',
            ha='center', fontsize=11.5, fontweight='bold', color='#1F3864')
    ax.text(675, 505, 'Unfiltered accuracy 92.31% \u2192 B(i)-filtered 60.47% \u2192 '
                      'age-residualized 48.8%  (\u221211.7 pp, demographic confounding)',
            ha='center', fontsize=10.5, color='#C00000', fontweight='bold')
    ax.plot([150, 1200], [500, 500], color='#1F3864', lw=0.8, ls=(0, (4, 4)))

    # 中排 5 框
    y = 300; w = 225; h = 140; gap = 30; x0 = 40
    xs = [x0 + i * (w + gap) for i in range(5)]
    box(xs[0], y, w, h, 'Preprocessing', 'Pharma / xenobiotic\nfiltering\n\u2192 1,966 B(i)\nlevel metabolites')
    box(xs[1], y, w, h, 'STEM', 'Ordinal pattern\ndiscovery\nP4 (n=642), P20 (n=371)\n1,013 metabolites')
    box(xs[2], y, w, h, 'Random Forest', 'Age-residualized\nbiomarker panel\n48.8% \u00b1 13.1%\n(5\u00d710-fold CV)')
    box(xs[3], y, w, h, 'NMF', 'Molecular subtyping\n(k=2)\nn=59 / 25')
    box(xs[4], y, w, h, 'Integration', '2-metabolite\n3-way key panel\n(STEM \u2229 RF \u2229 NMF)')
    for i in range(4):
        arrow(xs[i] + w + 4, y + h / 2, xs[i + 1] - 6)

    # 底部
    yb = 60
    ax.text(80, yb + 60, 'Raw LC-MS\n(4,357 features)', ha='center', fontsize=9.5, color='#1F3864')
    ax.plot([230, 230], [yb, yb + 130], color='#2F5597', lw=1.4)
    ax.plot([230, 480], [yb + 130, yb + 130], color='#2F5597', lw=1.4)
    ax.plot([480, 480], [yb, yb + 130], color='#2F5597', lw=1.4)
    ax.text(480, yb + 60, 'Age-residualized\nB(i) features (1,966)', ha='center', fontsize=9.5, color='#1F3864')
    ax.plot([620, 620], [yb, yb + 130], color='#2F5597', lw=1.4)
    ax.plot([620, 900], [yb + 130, yb + 130], color='#2F5597', lw=1.4)
    ax.plot([900, 900], [yb, yb + 130], color='#2F5597', lw=1.4)
    ax.text(900, yb + 60, 'Validation\nperm p=0.002 | nested CV 50.0%\nboot CI [34.5%, 54.8%]',
            ha='center', fontsize=9.2, color='#1F3864')
    ax.plot([1040, 1040], [yb, yb + 130], color='#2F5597', lw=1.4)
    ax.plot([1040, 1250], [yb + 130, yb + 130], color='#2F5597', lw=1.4)
    ax.plot([1250, 1250], [yb, yb + 130], color='#2F5597', lw=1.4)
    ax.text(1250, yb + 60, 'Validated\nbiomarkers', ha='center', fontsize=9.5, color='#1F3864', fontweight='bold')
    save(fig, 'Figure1.png')


# ============================================================ Figure 2
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    stages = ['HC', 'SCD', 'MCI', 'AD']

    def profile_curve(ax, ys, title, n, pval, fdr, note):
        x = np.arange(4)
        xs = np.linspace(0, 3, 200)
        ysm = np.interp(xs, x, ys)
        ax.plot(xs, ysm, color='#2F5597', lw=2.6)
        ax.scatter(x, ys, color='#C00000', zorder=5, s=46)
        ax.set_xticks(x); ax.set_xticklabels(stages)
        ax.set_ylim(-0.15, 1.25)
        ax.set_ylabel('Normalized abundance', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.text(0.03, 0.88, f'n = {n}\nP = {pval}\nFDR = {fdr}', transform=ax.transAxes,
                fontsize=9.5, va='top', color='#1F3864')
        ax.text(0.97, 0.02, note, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8.8, color='#555555')
        ax.axhline(0, color='#BBBBBB', lw=0.8)

    profile_curve(axes[0], [0.10, 0.12, 1.00, 0.30], 'Profile 4 \u2014 MCI-peak pattern',
                  642, '0.001', '0.025', 'baseline \u2192 peak at MCI \u2192 decline in AD')
    profile_curve(axes[1], [0.05, 1.00, 0.95, 0.35], 'Profile 20 \u2014 Early-rise pattern',
                  371, '0.001', '0.025', 'rise at SCD \u2192 plateau through MCI \u2192 decline in AD')

    ax = axes[2]
    ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    # 漏斗
    ax.add_patch(FancyBboxPatch((0.6, 7.2), 3.4, 1.6, boxstyle='round,pad=0.15', fc='#EAF1F8', ec='#2F5597'))
    ax.text(2.3, 8.0, '1,966 age-residualized\nB(i) metabolites', ha='center', fontsize=9.5, color='#1F3864')
    ax.plot([4.4, 6.4], [8.0, 8.0], color='#2F5597', lw=1.6)
    ax.add_patch(FancyBboxPatch((6.4, 6.4), 3.2, 3.2, boxstyle='round,pad=0.15', fc='#FBE5D6', ec='#C00000'))
    ax.text(8.0, 8.5, '2 significant\nprofiles', ha='center', fontsize=10.5, fontweight='bold', color='#C00000')
    ax.text(8.0, 7.2, '1,013 metabolites (51.5%)', ha='center', fontsize=9, color='#C00000')
    ax.plot([6.2, 6.2], [6.2, 4.3], color='#2F5597', lw=1.6)
    ax.text(6.5, 5.0, 'Enriched profiles:', fontsize=9.5, va='center', color='#1F3864')
    rows = [('Profile 4 (MCI peak)', '642'), ('Profile 20 (early rise)', '371'),
            ('Profile 22 (monotonic)', '0 (P=1.0)')]
    for i, (k, v) in enumerate(rows):
        ax.add_patch(FancyBboxPatch((1.0, 3.4 - i * 1.15), 8.2, 0.9, boxstyle='round,pad=0.12',
                                    fc='white', ec='#2F5597', lw=1.1))
        ax.text(1.6, 3.85 - i * 1.15, k, fontsize=9, va='center', color='#1F3864')
        ax.text(8.6, 3.85 - i * 1.15, v, fontsize=9, va='center', ha='right',
                color='#C00000' if i < 2 else '#888888', fontweight='bold' if i < 2 else 'normal')
    save(fig, 'Figure2.png')


# ============================================================ Figure 3
def fig3():
    corr = recomp['corrected']
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4))

    # A: MDG top10
    top10 = corr['mdg_top10']
    names = [m['name'] for m in top10][::-1]
    vals = [m['mdg'] for m in top10][::-1]
    ax = axes[0]
    short = [n if len(n) <= 24 else n[:22] + '...' for n in names]
    ax.barh(short, vals, color='#4C72B0')
    ax.set_xlabel('Mean Decrease Gini (normalized)', fontsize=9)
    ax.set_title('Top 10 biomarkers by MDG', fontsize=10.5, fontweight='bold')
    ax.tick_params(axis='y', labelsize=7.6)

    # B: ROC
    ax = axes[1]
    for g in GROUP_ORDER:
        pts = corr['roc_points'][g]
        ax.plot(pts['fpr'], pts['tpr'], lw=2.2, color=COLORS[g],
                label=f'{LABELS[g]} (AUC={corr["auc_ovr_per_class"][g]:.2f})')
    ax.plot([0, 1], [0, 1], ls='--', color='#888888', lw=1.0)
    ax.set_xlabel('False positive rate', fontsize=9)
    ax.set_ylabel('True positive rate', fontsize=9)
    ax.set_title('One-vs-rest ROC (age-residualized)', fontsize=10.5, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right', frameon=False)

    # C: CM
    ax = axes[2]
    cm = np.array(corr['cm'])
    ax.imshow(cm, cmap='Blues')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else '#1F3864')
    sens = corr['sensitivity']
    ax.set_xticks(range(4)); ax.set_xticklabels([LABELS[g] for g in GROUP_ORDER])
    ax.set_yticks(range(4)); ax.set_yticklabels([LABELS[g] for g in GROUP_ORDER])
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('True', fontsize=9)
    ax.set_title('Confusion matrix (50-fold cumulative)', fontsize=10.5, fontweight='bold')
    sens_txt = '  |  '.join(f'{LABELS[g]} {int(round(s * 100))}%' for g, s in sens.items())
    ax.text(0.5, -0.14, f'Sensitivity: {sens_txt}', transform=ax.transAxes, ha='center',
            fontsize=8.5, color='#1F3864')
    save(fig, 'Figure3.png')


# ============================================================ Figure 4
def fig4():
    k2 = a8['nmf_k_comparison']['2']
    subtypes = np.array(k2['subtypes'])
    da0 = k2['disease_association']['subtype_0']
    da1 = k2['disease_association']['subtype_1']

    # PCA 重算
    annot, expr = load_expression_matrix()
    crf = load_clinical_data()
    _, expr_bi = filter_by_level(annot, expr, 'B(i)')
    expr_log = log_transform(expr_bi)
    expr_resid = age_residualization(expr_log, crf)
    samples, groups = get_sample_groups(expr_resid, crf)
    X = expr_resid[samples].values.T
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xs)
    pc1, pc2 = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4))

    # A: PCA
    ax = axes[0]
    for s in [0, 1]:
        m = subtypes == s
        ax.scatter(coords[m, 0], coords[m, 1], s=34, alpha=0.85, color=SUB_COLORS[s],
                   label=f'Subtype {s+1} (n={m.sum()})', edgecolor='white', lw=0.5)
    ax.set_xlabel(f'PC1 ({pc1 * 100:.1f}%)', fontsize=9)
    ax.set_ylabel(f'PC2 ({pc2 * 100:.1f}%)', fontsize=9)
    ax.set_title('PCA by NMF subtype (k=2)', fontsize=10.5, fontweight='bold')
    ax.legend(fontsize=8.5, loc='best', frameon=False)

    # B: subtype x disease
    ax = axes[1]
    counts = np.array([[5, 5], [16, 6], [20, 3], [18, 11]], dtype=float)
    ax.imshow(counts, cmap='YlGnBu')
    totals = counts.sum(axis=1)
    for i in range(4):
        for j in range(2):
            v = counts[i, j]
            pct = v / totals[i] * 100 if totals[i] else 0
            ax.text(j, i, f'{int(v)}\n({pct:.0f}%)', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if v > counts.max() / 2 else '#1F3864')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Subtype 1\n(n=59)', 'Subtype 2\n(n=25)'], fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels([LABELS[g] for g in GROUP_ORDER], fontsize=9)
    ax.set_title('Subtype \u00d7 disease', fontsize=10.5, fontweight='bold')

    # C: disease composition stacked
    ax = axes[2]
    order = ['CG', 'SCD', 'MCI', 'AD']
    bottom0 = np.array([da0['disease_dist'][g] for g in order])
    bottom1 = np.array([da1['disease_dist'][g] for g in order])
    x = np.arange(2)
    for i, g in enumerate(order):
        ax.bar(0, bottom0[i], bottom=bottom0[:i].sum(), color=COLORS[g],
               label=LABELS[g] if i == 0 else None, width=0.5)
        ax.bar(1, bottom1[i], bottom=bottom1[:i].sum(), color=COLORS[g], width=0.5)
    ax.set_xticks(x); ax.set_xticklabels(['Subtype 1\n(n=59)', 'Subtype 2\n(n=25)'], fontsize=9)
    ax.set_ylabel('Proportion of subtype', fontsize=9)
    ax.set_title('Disease composition', fontsize=10.5, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    save(fig, 'Figure4.png')


# ============================================================ Figure 6 (S6 分解)
def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    # A: 两阶段精度
    ax = axes[0]
    stages = ['B(i)-filtered\n(uncorrected)', 'Age-residualized']
    accs = [60.47, 48.8]
    bars = ax.bar(stages, accs, color=['#4C72B0', '#C44E52'], width=0.5)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 1.2, f'{a:.2f}%'.rstrip('0').rstrip('.') + '%',
                ha='center', fontsize=10.5, fontweight='bold', color='#1F3864')
    ax.axhline(25, ls='--', color='#888888', lw=1.2)
    ax.text(1.45, 27, 'chance = 25%', fontsize=8.5, color='#555555')
    ax.annotate('\u221211.7 pp\n(demographic\nconfounding)', xy=(1, 48.8), xytext=(0.42, 40),
                fontsize=9, color='#C00000', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.5))
    ax.set_ylim(0, 70)
    ax.set_ylabel('4-class accuracy (%)', fontsize=9)
    ax.set_title('Accuracy: demographic confounding', fontsize=10.5, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8.5)

    # B: per-class sensitivity
    ax = axes[1]
    unc = [34, 48, 36, 93]
    cor = [6, 28, 50, 78]
    x = np.arange(4)
    w = 0.35
    ax.bar(x - w / 2, unc, w, label='Uncorrected', color='#4C72B0')
    ax.bar(x + w / 2, cor, w, label='Age-residualized', color='#C44E52')
    for i in range(4):
        ax.text(x[i] - w / 2, unc[i] + 2, f'{unc[i]}%', ha='center', fontsize=8.5, color='#1F3864')
        ax.text(x[i] + w / 2, cor[i] + 2, f'{cor[i]}%', ha='center', fontsize=8.5, color='#1F3864')
    ax.set_xticks(x); ax.set_xticklabels([LABELS[g] for g in GROUP_ORDER])
    ax.set_ylim(0, 105)
    ax.set_ylabel('Sensitivity (%)', fontsize=9)
    ax.set_title('Per-class sensitivity', fontsize=10.5, fontweight='bold')
    ax.legend(fontsize=8.5, frameon=False)
    save(fig, 'Figure6.png')
    # S6 独立版（docx 嵌入用）
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    bars = ax.bar(stages, accs, color=['#4C72B0', '#C44E52'], width=0.5)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 1.2, f'{a:.2f}%'.rstrip('0').rstrip('.') + '%',
                ha='center', fontsize=10.5, fontweight='bold', color='#1F3864')
    ax.axhline(25, ls='--', color='#888888', lw=1.2)
    ax.annotate('\u221211.7 pp', xy=(1, 48.8), xytext=(0.40, 42), fontsize=9, color='#C00000',
                fontweight='bold', arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.5))
    ax.set_ylim(0, 70); ax.set_ylabel('4-class accuracy (%)', fontsize=9)
    ax.set_title('(A) Accuracy', fontsize=10.5, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8.5)
    ax = axes[1]
    x = np.arange(4); w = 0.35
    ax.bar(x - w / 2, unc, w, label='Uncorrected', color='#4C72B0')
    ax.bar(x + w / 2, cor, w, label='Age-residualized', color='#C44E52')
    for i in range(4):
        ax.text(x[i] - w / 2, unc[i] + 2, f'{unc[i]}%', ha='center', fontsize=8.5, color='#1F3864')
        ax.text(x[i] + w / 2, cor[i] + 2, f'{cor[i]}%', ha='center', fontsize=8.5, color='#1F3864')
    ax.set_xticks(x); ax.set_xticklabels([LABELS[g] for g in GROUP_ORDER])
    ax.set_ylim(0, 105); ax.set_ylabel('Sensitivity (%)', fontsize=9)
    ax.set_title('(B) Per-class sensitivity', fontsize=10.5, fontweight='bold')
    ax.legend(fontsize=8.5, frameon=False)
    fig.savefig(os.path.join(SUP, 'Fig_S6_Decomposition.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('saved', os.path.join(SUP, 'Fig_S6_Decomposition.png'))


# ============================================================ Graphical Abstract
def graph_abs():
    fig, ax = plt.subplots(figsize=(13.2, 9.3))
    ax.set_xlim(0, 1320); ax.set_ylim(0, 930); ax.axis('off')

    # 标题
    ax.text(660, 900, 'Three-stage integrated metabolomics pipeline reveals a 2-metabolite\n'
                      'AD-discriminatory key panel across the AD continuum',
            ha='center', fontsize=15, fontweight='bold', color='#1F3864')

    # INPUT 条
    ax.add_patch(FancyBboxPatch((120, 780), 1080, 85, boxstyle='round,pad=8', fc='#EAF1F8', ec='#2F5597', lw=1.5))
    ax.text(140, 826, 'INPUT DATA', fontsize=11, fontweight='bold', color='#1F3864')
    ax.text(140, 800, '84 fecal LC-MS samples | 4 AD stages (HC=10, SCD=22, MCI=23, AD=29)\n'
                      'March 2022 \u2013 December 2023 | Shanghai, China | MetaboLights',
            fontsize=9.5, color='#1F3864', va='top')

    # 4 阶段框
    xs = [90, 385, 680, 975]; w = 260; y = 560; h = 185
    for i, (title, body) in enumerate([
        ('Stage 0\nPharmaceutical\nFiltering', '\u2192 1,966\nB(i)-level'),
        ('Stage 1\nSTEM Pattern\nDiscovery', '2 significant\nprofiles (1,013)'),
        ('Stage 2\nRandom Forest\n(Age-resid.)', '10 endogenous\nbiomarkers (48.8%)'),
        ('Stage 3\nNMF Subtype\n(k=2)', '2 subtypes\n(n=59 / 25)'),
    ]):
        ax.add_patch(FancyBboxPatch((xs[i], y), w, h, boxstyle='round,pad=8', fc='#FDF6EC', ec='#C55A11', lw=1.6))
        ax.text(xs[i] + w / 2, y + h - 38, title, ha='center', fontsize=10, fontweight='bold',
                color='#843C0C', va='top')
        ax.text(xs[i] + w / 2, y + 45, body, ha='center', fontsize=10, color='#1F3864')
        if i < 3:
            ax.add_patch(FancyArrowPatch((xs[i] + w + 6, y + h / 2), (xs[i + 1] - 8, y + h / 2),
                                         arrowstyle='-|>', mutation_scale=26, lw=2.2, color='#2F5597'))

    # KEY PANEL
    ax.add_patch(FancyBboxPatch((140, 300), 460, 180, boxstyle='round,pad=8', fc='#FFF2CC', ec='#BF9000', lw=2.0))
    ax.text(370, 448, '\u2605 KEY PANEL (3-way intersection)', ha='center', fontsize=12, fontweight='bold', color='#7F6000')
    ax.text(370, 420, '9,12,13-trihydroxyoctadec-15-enoic acid', ha='center', fontsize=10.5, color='#1F3864')
    ax.text(370, 402, '(oxygenated fatty acid)', ha='center', fontsize=9, color='#555555')
    ax.text(370, 368, 'N-oleoyl-L-serine', ha='center', fontsize=10.5, color='#1F3864')
    ax.text(370, 350, '(endocannabinoid-like lipid)', ha='center', fontsize=9, color='#555555')
    ax.text(370, 322, 'Cross-validated by STEM + RF + NMF', ha='center', fontsize=9.5, color='#7F6000')

    # CLINICAL APPLICATIONS
    ax.add_patch(FancyBboxPatch((720, 300), 460, 180, boxstyle='round,pad=8', fc='#E2EFDA', ec='#538135', lw=1.8))
    ax.text(950, 448, 'CLINICAL APPLICATIONS', ha='center', fontsize=11.5, fontweight='bold', color='#375623')
    apps = ['Early SCD/MCI detection (preclinical AD screening)',
            'Molecular subtyping (Subtype 1: 62% AD / 87% MCI)',
            'Therapeutic target discovery (oxygenated FA + endocannabinoid)']
    for k, a in enumerate(apps):
        ax.text(950, 420 - k * 40, a, ha='center', fontsize=9.5, color='#375623')

    # 底部
    ax.text(660, 210, 'Accuracy: B(i)-filtered 60.47% \u2192 age-residualized 48.8%  (\u221211.7 pp)',
            ha='center', fontsize=11.5, fontweight='bold', color='#C00000')
    ax.text(660, 165, 'HC (n=10)   SCD (n=22)   MCI (n=23)   AD (n=29)',
            ha='center', fontsize=9.5, color='#1F3864')
    ax.add_patch(FancyBboxPatch((120, 90), 1080, 55, boxstyle='round,pad=8', fc='#F2F2F2', ec='#888888', lw=1.0))
    ax.text(660, 118, 'Permutation p = 0.002 | bootstrap 95% CI [34.5%, 54.8%] | nested CV 50.0% | '
                      'NMF silhouette 0.051, cophenetic 0.732',
            ha='center', fontsize=9, color='#333333')
    save(fig, 'Graphical_Abstract.png')


def fig5():
    """Figure 5: Three-method metabolite set integration (real intersection counts).
    Data: STEM 1,013; RF top 50; NMF top 80; pairwise 31/42/3; 3-way 2 (key panel)."""
    fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=600)
    ax.set_xlim(0, 1320); ax.set_ylim(0, 700)
    ax.axis('off')

    # circles: STEM left, NMF right, RF top
    from matplotlib.patches import Circle
    ax.add_patch(Circle((380, 340), 235, fc='#DEEBF7', ec='#2E75B6', lw=2.0, alpha=0.55))
    ax.add_patch(Circle((650, 340), 235, fc='#FDE9D9', ec='#C55A11', lw=2.0, alpha=0.55))
    ax.add_patch(Circle((515, 560), 235, fc='#E2EFDA', ec='#538135', lw=2.0, alpha=0.55))

    # region counts
    regs = [
        ('STEM (1,013)', 300, 120, '#1F3864', 14, True),
        ('NMF (80)', 730, 120, '#833C00', 14, True),
        ('RF (50)', 515, 760, '#375623', 14, True),
        ('938', 285, 340, '#1F3864', 15, False),
        ('33', 725, 340, '#833C00', 15, False),
        ('14', 515, 660, '#375623', 15, False),
        ('31', 350, 500, '#1F3864', 17, False),
        ('42', 450, 300, '#833C00', 17, False),
        ('3', 670, 500, '#375623', 17, False),
        ('2', 515, 380, '#000000', 22, False),
    ]
    for txt, x, y, c, fs, bold in regs:
        ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'bold', color=c)

    # key panel box (top-right)
    ax.add_patch(FancyBboxPatch((860, 380), 430, 250, boxstyle='round,pad=10',
                                fc='#FFF2CC', ec='#BF8F00', lw=1.8))
    ax.text(1075, 610, 'Key panel', ha='center', fontsize=13, fontweight='bold', color='#7F6000')
    ax.text(1075, 582, '(3-way intersection, n=2)', ha='center', fontsize=10, color='#7F6000')
    ax.text(1075, 530, '9,12,13-trihydroxyoctadec-15-enoic acid', ha='center', fontsize=10.5, color='#1F3864')
    ax.text(1075, 505, '(oxygenated fatty acid)', ha='center', fontsize=9, color='#555555')
    ax.text(1075, 470, 'N-oleoyl-L-serine', ha='center', fontsize=10.5, color='#1F3864')
    ax.text(1075, 445, '(endocannabinoid-like lipid)', ha='center', fontsize=9, color='#555555')
    ax.text(1075, 415, 'Highest confidence', ha='center', fontsize=10, fontweight='bold', color='#BF8F00')
    ax.text(1075, 395, 'STEM + RF + NMF \u2014 cross-method validation', ha='center', fontsize=9, color='#555555')

    ax.set_title('Figure 5. Three-method metabolite set integration',
                 fontsize=14, fontweight='bold', pad=15, color='#1F3864')
    save(fig, 'Figure5.png')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    graph_abs()
    print('ALL MAIN FIGURES DONE')
