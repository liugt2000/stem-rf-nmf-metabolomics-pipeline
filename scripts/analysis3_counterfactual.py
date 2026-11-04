# -*- coding: utf-8 -*-
"""
分析3：反事实对比（未校正会得出什么错误结论）
强化文章核心卖点，让编辑眼前一亮

对比维度：
1. 未校正（全部4357特征，无年龄残差化）vs 校正后（B(i) 1966特征，年龄残差化）
2. 分类准确率对比
3. Top生物标志物的药物/外源物比例
4. 富集通路差异
5. NMF亚型与年龄/疾病的关联
6. 混淆矩阵对比
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import NMF
from scipy import stats
import sys
import json
import os

sys.path.insert(0, r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析')
from analysis_utils import (load_expression_matrix, load_clinical_data,
                            filter_by_level, log_transform, age_residualization,
                            get_sample_groups, GROUP_ORDER, GROUP_STAGE)

# ========== 配置 ==========
N_ESTIMATORS = 500
RANDOM_STATE = 42
OUTPUT_DIR = r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析\Analysis_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_rf_full(X, y, n_estimators=N_ESTIMATORS):
    """运行RF 5折CV，返回完整结果"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    y_true_all = []
    y_pred_all = []
    y_proba_all = []
    importances = np.zeros(X.shape[1])
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features='sqrt',
            # 不使用class_weight='balanced'，和论文一致
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        importances += clf.feature_importances_
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_proba_all.append(y_proba)
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    accuracy = accuracy_score(y_true_all, y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=GROUP_ORDER)
    importances /= 5
    
    return {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'importances': importances,
    }


def run_nmf_analysis(X, n_components=2):
    """运行NMF，返回亚型分配和样本权重"""
    X_nonneg = X - X.min() + 1e-10
    model = NMF(n_components=n_components, init='nndsvda', 
                max_iter=1000, random_state=RANDOM_STATE)
    W = model.fit_transform(X_nonneg)
    subtypes = np.argmax(W, axis=1)
    return subtypes, W


def classify_metabolite_type(metab_row):
    """分类代谢物类型：内源性 vs 可能药物/外源物"""
    # 基于level和注释判断
    level = str(metab_row.get('level', ''))
    name = str(metab_row.get('Metabolite', '')).lower()
    kegg = str(metab_row.get('KEGG Compound ID', ''))
    
    # B(ii)级更可能包含外源物
    if level == 'B(ii)':
        # 检查是否有KEGG注释（有KEGG的更可能是内源性）
        if kegg and kegg != 'nan' and not kegg.startswith('C'):
            return 'B(ii)_with_KEGG'
        return 'B(ii)_likely_xenobiotic'
    
    # B(i)级更可能是内源性
    if level == 'B(i)':
        return 'B(i)_endogenous_like'
    
    return 'unknown'


def analyze_top_metabolites(importances, annot, n_top=50):
    """分析top-N代谢物的类型分布"""
    top_idx = np.argsort(importances)[-n_top:][::-1]
    top_metab_ids = annot.iloc[top_idx]['metab_id'].values
    
    # 获取这些代谢物的注释
    top_annot = annot.iloc[top_idx].copy()
    
    # 分类
    type_counts = {'B(i)_endogenous_like': 0, 'B(ii)_likely_xenobiotic': 0, 
                   'B(ii)_with_KEGG': 0, 'unknown': 0}
    level_counts = {'B(i)': 0, 'B(ii)': 0}
    mode_counts = {'pos': 0, 'neg': 0}
    kegg_present = 0
    
    for _, row in top_annot.iterrows():
        mtype = classify_metabolite_type(row)
        type_counts[mtype] = type_counts.get(mtype, 0) + 1
        
        level = str(row.get('level', ''))
        if level in level_counts:
            level_counts[level] += 1
        
        mode = str(row.get('Mode', ''))
        if mode in mode_counts:
            mode_counts[mode] += 1
        
        kegg = str(row.get('KEGG Compound ID', ''))
        if kegg and kegg != 'nan' and kegg.startswith('C'):
            kegg_present += 1
    
    total = len(top_annot)
    result = {
        'n_top': n_top,
        'type_counts': type_counts,
        'type_percentages': {k: v/total*100 for k, v in type_counts.items()},
        'level_counts': level_counts,
        'level_percentages': {k: v/total*100 for k, v in level_counts.items()},
        'mode_counts': mode_counts,
        'kegg_present': kegg_present,
        'kegg_percentage': kegg_present/total*100,
        'top_metabolite_names': top_annot['Metabolite'].tolist()[:20],
    }
    
    return result


def nmf_subtype_correlation(subtypes, crf, samples):
    """分析NMF亚型与年龄、疾病阶段的关联"""
    # 获取样本的临床数据
    sample_data = crf.loc[samples].copy()
    sample_data['subtype'] = subtypes
    
    # 亚型与年龄的关联（t检验）
    subtype0_age = sample_data[sample_data['subtype'] == 0]['Age'].values
    subtype1_age = sample_data[sample_data['subtype'] == 1]['Age'].values
    
    if len(subtype0_age) > 1 and len(subtype1_age) > 1:
        t_stat, p_val = stats.ttest_ind(subtype0_age, subtype1_age)
        age_diff = np.mean(subtype0_age) - np.mean(subtype1_age)
    else:
        t_stat, p_val = np.nan, np.nan
        age_diff = np.nan
    
    # 亚型与疾病阶段的关联（Cramer's V或卡方）
    contingency = pd.crosstab(sample_data['subtype'], sample_data['Group'])
    try:
        chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
    except:
        chi2, p_chi2 = np.nan, np.nan
    
    # 各亚型的疾病组成
    subtype_composition = {}
    for st in [0, 1]:
        st_data = sample_data[sample_data['subtype'] == st]
        composition = st_data['Group'].value_counts(normalize=True).to_dict()
        subtype_composition[f'subtype_{st}'] = {
            'n': len(st_data),
            'mean_age': float(st_data['Age'].mean()),
            'age_std': float(st_data['Age'].std()),
            'disease_composition': {k: float(v) for k, v in composition.items()},
        }
    
    return {
        'age_t_statistic': float(t_stat) if not np.isnan(t_stat) else None,
        'age_p_value': float(p_val) if not np.isnan(p_val) else None,
        'age_mean_diff': float(age_diff) if not np.isnan(age_diff) else None,
        'disease_chi2': float(chi2) if not np.isnan(chi2) else None,
        'disease_p_value': float(p_chi2) if not np.isnan(p_chi2) else None,
        'subtype_composition': subtype_composition,
    }


def main():
    print("=" * 70)
    print("分析3：反事实对比（未校正会得出什么错误结论）")
    print("=" * 70)
    
    # ========== 1. 加载数据 ==========
    print("\n[1/5] 加载数据...")
    annot, expr = load_expression_matrix()
    crf = load_clinical_data()
    
    # ========== 2. 准备两种数据版本 ==========
    print("\n[2/5] 准备两种数据版本...")
    
    # 先过滤B(i)级（和论文的Unfiltered 1966一致）
    annot_bi, expr_bi = filter_by_level(annot, expr, 'B(i)')
    expr_log_bi = log_transform(expr_bi)
    
    # 版本A：未校正（B(i)级1966特征，仅log转换，无年龄残差化）
    print("\n  版本A：未校正（B(i)级1966特征，无年龄残差化）")
    samples_all, groups_all = get_sample_groups(expr_log_bi, crf)
    X_uncorrected = expr_log_bi[samples_all].values.T
    y_all = groups_all
    print(f"    矩阵: {X_uncorrected.shape}")
    
    # 版本B：校正后（B(i)级1966特征，年龄残差化）
    print("\n  版本B：校正后（B(i)级1966特征，年龄残差化）")
    expr_resid = age_residualization(expr_log_bi, crf)
    samples_corr, groups_corr = get_sample_groups(expr_resid, crf)
    X_corrected = expr_resid[samples_corr].values.T
    y_corr = groups_corr
    print(f"    矩阵: {X_corrected.shape}")
    
    # ========== 3. RF分类对比 ==========
    print("\n[3/5] RF分类对比...")
    
    print("\n  未校正模型...")
    rf_uncorrected = run_rf_full(X_uncorrected, y_all)
    print(f"    准确率: {rf_uncorrected['accuracy']:.1%}")
    
    print("\n  校正后模型...")
    rf_corrected = run_rf_full(X_corrected, y_corr)
    print(f"    准确率: {rf_corrected['accuracy']:.1%}")
    
    acc_diff = rf_uncorrected['accuracy'] - rf_corrected['accuracy']
    print(f"\n  准确率差异: {acc_diff*100:.1f} 个百分点")
    
    # ========== 4. Top生物标志物对比 ==========
    print("\n[4/5] Top生物标志物对比...")
    
    # 未校正的top-50
    print("\n  未校正模型Top-50代谢物分析...")
    top_uncorrected = analyze_top_metabolites(rf_uncorrected['importances'], annot, n_top=50)
    print(f"    B(i)级比例: {top_uncorrected['level_percentages'].get('B(i)', 0):.1f}%")
    print(f"    B(ii)级比例: {top_uncorrected['level_percentages'].get('B(ii)', 0):.1f}%")
    print(f"    有KEGG注释比例: {top_uncorrected['kegg_percentage']:.1f}%")
    print(f"    Top-10名称: {top_uncorrected['top_metabolite_names'][:10]}")
    
    # 校正后的top-50
    print("\n  校正后模型Top-50代谢物分析...")
    top_corrected = analyze_top_metabolites(rf_corrected['importances'], annot_bi, n_top=50)
    print(f"    B(i)级比例: {top_corrected['level_percentages'].get('B(i)', 0):.1f}%")
    print(f"    有KEGG注释比例: {top_corrected['kegg_percentage']:.1f}%")
    print(f"    Top-10名称: {top_corrected['top_metabolite_names'][:10]}")
    
    # Top-10重叠率
    top10_uncorr = set(annot.iloc[np.argsort(rf_uncorrected['importances'])[-10:][::-1]]['metab_id'].values)
    top10_corr = set(annot_bi.iloc[np.argsort(rf_corrected['importances'])[-10:][::-1]]['metab_id'].values)
    overlap_10 = len(top10_uncorr & top10_corr)
    print(f"\n  Top-10重叠: {overlap_10}/10")
    
    # ========== 5. NMF亚型对比 ==========
    print("\n[5/5] NMF亚型对比...")
    
    print("\n  未校正数据NMF...")
    subtypes_uncorr, W_uncorr = run_nmf_analysis(X_uncorrected, n_components=2)
    nmf_uncorr = nmf_subtype_correlation(subtypes_uncorr, crf, samples_all)
    print(f"    亚型与年龄关联 p={nmf_uncorr['age_p_value']:.4f}")
    print(f"    亚型与疾病关联 p={nmf_uncorr['disease_p_value']:.4f}")
    for st, comp in nmf_uncorr['subtype_composition'].items():
        print(f"    {st}: n={comp['n']}, 平均年龄={comp['mean_age']:.1f}, "
              f"疾病组成={ {k: f'{v:.0%}' for k, v in comp['disease_composition'].items()} }")
    
    print("\n  校正后数据NMF...")
    subtypes_corr, W_corr = run_nmf_analysis(X_corrected, n_components=2)
    nmf_corr = nmf_subtype_correlation(subtypes_corr, crf, samples_corr)
    print(f"    亚型与年龄关联 p={nmf_corr['age_p_value']:.4f}")
    print(f"    亚型与疾病关联 p={nmf_corr['disease_p_value']:.4f}")
    for st, comp in nmf_corr['subtype_composition'].items():
        print(f"    {st}: n={comp['n']}, 平均年龄={comp['mean_age']:.1f}, "
              f"疾病组成={ {k: f'{v:.0%}' for k, v in comp['disease_composition'].items()} }")
    
    # ========== 保存结果 ==========
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    results = {
        'uncorrected': {
            'accuracy': rf_uncorrected['accuracy'],
            'confusion_matrix': rf_uncorrected['confusion_matrix'],
            'top50_analysis': top_uncorrected,
            'nmf_analysis': nmf_uncorr,
        },
        'corrected': {
            'accuracy': rf_corrected['accuracy'],
            'confusion_matrix': rf_corrected['confusion_matrix'],
            'top50_analysis': top_corrected,
            'nmf_analysis': nmf_corr,
        },
        'comparison': {
            'accuracy_difference_pp': acc_diff * 100,
            'top10_overlap': overlap_10,
            'uncorrected_Bii_percentage': top_uncorrected['level_percentages'].get('B(ii)', 0),
            'corrected_Bii_percentage': top_corrected['level_percentages'].get('B(ii)', 0),
        }
    }
    
    with open(f'{OUTPUT_DIR}\\analysis3_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # 关键结论
    print("\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)
    
    print(f"\n1. 准确率膨胀:")
    print(f"   未校正: {rf_uncorrected['accuracy']:.1%}")
    print(f"   校正后: {rf_corrected['accuracy']:.1%}")
    print(f"   膨胀: {acc_diff*100:.1f} 个百分点")
    
    print(f"\n2. 生物标志物污染:")
    print(f"   未校正Top-50中B(ii)级比例: {top_uncorrected['level_percentages'].get('B(ii)', 0):.1f}%")
    print(f"   校正后Top-50中B(ii)级比例: {top_corrected['level_percentages'].get('B(ii)', 0):.1f}%")
    print(f"   Top-10重叠率: {overlap_10}/10")
    
    print(f"\n3. NMF亚型驱动因素:")
    print(f"   未校正: 年龄关联 p={nmf_uncorr['age_p_value']:.4f}, 疾病关联 p={nmf_uncorr['disease_p_value']:.4f}")
    print(f"   校正后: 年龄关联 p={nmf_corr['age_p_value']:.4f}, 疾病关联 p={nmf_corr['disease_p_value']:.4f}")
    
    if nmf_uncorr['age_p_value'] < 0.05 and nmf_corr['age_p_value'] > 0.05:
        print(f"\n✅ 未校正NMF亚型主要由年龄驱动，校正后年龄效应消失")
        print(f"   这证明不做混杂控制会把年龄相关代谢变化误认为疾病亚型")
    
    print(f"\n结果已保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
