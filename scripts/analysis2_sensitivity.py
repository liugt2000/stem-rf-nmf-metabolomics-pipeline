# -*- coding: utf-8 -*-
"""
分析2：年龄残差化敏感性分析（3种方法对比）
把"方法选择依据不足"从弱点变加分项

三种方法：
1. 线性回归残差化（当前方法）
2. ComBat批次校正（年龄分箱后当批次）
3. 年龄分箱分层标准化

对比指标：
- 四分类准确率（5折CV）
- Top-10代谢物重叠率（Jaccard）
- NMF亚型一致性（Adjusted Rand Index）
- 每类灵敏度
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score
import sys
import json
import os

sys.path.insert(0, r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析')
from analysis_utils import (load_expression_matrix, load_clinical_data,
                            filter_by_level, log_transform, age_residualization,
                            combat_adjust, age_binning_standardize,
                            get_sample_groups, GROUP_ORDER)

# ========== 配置 ==========
N_ESTIMATORS = 500
RANDOM_STATE = 42
OUTPUT_DIR = r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析\Analysis_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_rf_with_feature_importance(X, y, n_estimators=N_ESTIMATORS):
    """运行RF 5折CV，返回准确率、混淆矩阵、特征重要度"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    y_true_all = []
    y_pred_all = []
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
        importances += clf.feature_importances_
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    accuracy = accuracy_score(y_true_all, y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=GROUP_ORDER)
    importances /= 5  # 平均
    
    return accuracy, cm, importances


def run_nmf(X, n_components=2, random_state=RANDOM_STATE):
    """运行NMF，返回样本亚型分配"""
    # NMF需要非负输入，将残差平移到非负
    X_nonneg = X - X.min() + 1e-10
    
    model = NMF(n_components=n_components, init='nndsvda', 
                max_iter=1000, random_state=random_state)
    W = model.fit_transform(X_nonneg)  # 样本 × 亚型
    
    # 分配到概率最大的亚型
    subtypes = np.argmax(W, axis=1)
    
    return subtypes, W


def jaccard_index(set1, set2):
    """计算两个集合的Jaccard指数"""
    set1 = set(set1)
    set2 = set(set2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def main():
    print("=" * 70)
    print("分析2：年龄残差化敏感性分析（3种方法对比）")
    print("=" * 70)
    
    # ========== 1. 加载和预处理数据 ==========
    print("\n[1/4] 加载数据...")
    annot, expr = load_expression_matrix()
    crf = load_clinical_data()
    
    # 过滤B(i)级
    annot_bi, expr_bi = filter_by_level(annot, expr, 'B(i)')
    
    # log转换
    expr_log = log_transform(expr_bi)
    
    # 代谢物ID列表（用于top-10对比）
    metabolite_ids = expr_bi.index.tolist()
    
    # ========== 2. 三种残差化方法 ==========
    print("\n[2/4] 三种年龄残差化方法...")
    
    # 方法1：线性回归残差化（当前方法）
    print("\n  方法1：线性回归残差化")
    expr_method1 = age_residualization(expr_log, crf, 
                                        covariates=['Age', 'Education_code', 'Gender_code'])
    
    # 方法2：ComBat批次校正
    print("\n  方法2：ComBat批次校正")
    expr_method2 = combat_adjust(expr_log, crf, batch_var='Age')
    
    # 方法3：年龄分箱分层标准化
    print("\n  方法3：年龄分箱分层标准化")
    expr_method3 = age_binning_standardize(expr_log, crf, n_bins=3)
    
    # 未校正（基准）
    print("\n  基准：未校正（仅log转换）")
    expr_uncorrected = expr_log.copy()
    
    # ========== 3. 对比分析 ==========
    print("\n[3/4] 对比分析...")
    
    methods = {
        'Uncorrected': expr_uncorrected,
        'Method1_LinearRegression': expr_method1,
        'Method2_ComBat': expr_method2,
        'Method3_AgeBinning': expr_method3,
    }
    
    results = {}
    all_importances = {}
    all_subtypes = {}
    
    for method_name, expr_data in methods.items():
        print(f"\n  --- {method_name} ---")
        
        # 获取样本和标签
        samples, groups = get_sample_groups(expr_data, crf)
        X = expr_data[samples].values.T
        y = groups
        
        # RF分类
        acc, cm, importances = run_rf_with_feature_importance(X, y)
        print(f"    准确率: {acc:.4f}")
        
        # 每类灵敏度
        sensitivities = {}
        for i, g in enumerate(GROUP_ORDER):
            if cm[i].sum() > 0:
                sens = cm[i, i] / cm[i].sum()
                sensitivities[g] = float(sens)
                print(f"    {g}灵敏度: {sens:.1%}")
        
        # Top-10代谢物
        top10_idx = np.argsort(importances)[-10:][::-1]
        top10_metabolites = [metabolite_ids[i] for i in top10_idx]
        print(f"    Top-10代谢物: {top10_metabolites[:5]}...")
        
        # NMF亚型
        subtypes, W = run_nmf(X, n_components=2)
        print(f"    NMF亚型分布: {dict(pd.Series(subtypes).value_counts())}")
        
        results[method_name] = {
            'accuracy': float(acc),
            'confusion_matrix': cm.tolist(),
            'sensitivities': sensitivities,
            'top10_metabolites': top10_metabolites,
        }
        all_importances[method_name] = importances
        all_subtypes[method_name] = subtypes
    
    # ========== 4. 方法间对比 ==========
    print("\n[4/4] 方法间对比...")
    
    comparison = {}
    
    # Top-10 Jaccard重叠率
    print("\n  Top-10代谢物Jaccard重叠率:")
    jaccard_matrix = {}
    method_names = list(methods.keys())
    for i, m1 in enumerate(method_names):
        jaccard_matrix[m1] = {}
        for j, m2 in enumerate(method_names):
            jac = jaccard_index(results[m1]['top10_metabolites'], 
                                results[m2]['top10_metabolites'])
            jaccard_matrix[m1][m2] = float(jac)
            if i < j:
                print(f"    {m1} vs {m2}: {jac:.3f}")
    
    comparison['top10_jaccard'] = jaccard_matrix
    
    # NMF亚型一致性（Adjusted Rand Index）
    print("\n  NMF亚型一致性（Adjusted Rand Index）:")
    ari_matrix = {}
    for i, m1 in enumerate(method_names):
        ari_matrix[m1] = {}
        for j, m2 in enumerate(method_names):
            ari = adjusted_rand_score(all_subtypes[m1], all_subtypes[m2])
            ari_matrix[m1][m2] = float(ari)
            if i < j:
                print(f"    {m1} vs {m2}: {ari:.3f}")
    
    comparison['nmf_ari'] = ari_matrix
    
    # 准确率对比
    print("\n  准确率对比:")
    for m in method_names:
        print(f"    {m}: {results[m]['accuracy']:.4f}")
    
    # ========== 保存结果 ==========
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    output = {
        'methods': results,
        'comparison': comparison,
        'method_names': method_names,
    }
    
    with open(f'{OUTPUT_DIR}\\analysis2_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # 保存关键对比表
    comparison_table = []
    for m in method_names:
        row = {
            'Method': m,
            'Accuracy': f"{results[m]['accuracy']:.1%}",
            'CG_sensitivity': f"{results[m]['sensitivities'].get('CG', 0):.1%}",
            'SCD_sensitivity': f"{results[m]['sensitivities'].get('SCD', 0):.1%}",
            'MCI_sensitivity': f"{results[m]['sensitivities'].get('MCI', 0):.1%}",
            'AD_sensitivity': f"{results[m]['sensitivities'].get('AD', 0):.1%}",
        }
        comparison_table.append(row)
    
    comparison_df = pd.DataFrame(comparison_table)
    comparison_df.to_csv(f'{OUTPUT_DIR}\\analysis2_comparison_table.csv', 
                         index=False, encoding='utf-8-sig')
    print("\n方法对比表:")
    print(comparison_df.to_string(index=False))
    
    # 关键结论
    print("\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)
    
    # 三种校正方法之间的一致性
    m1_vs_m2_jac = jaccard_matrix['Method1_LinearRegression']['Method2_ComBat']
    m1_vs_m3_jac = jaccard_matrix['Method1_LinearRegression']['Method3_AgeBinning']
    m2_vs_m3_jac = jaccard_matrix['Method2_ComBat']['Method3_AgeBinning']
    
    avg_jac = (m1_vs_m2_jac + m1_vs_m3_jac + m2_vs_m3_jac) / 3
    
    m1_vs_m2_ari = ari_matrix['Method1_LinearRegression']['Method2_ComBat']
    m1_vs_m3_ari = ari_matrix['Method1_LinearRegression']['Method3_AgeBinning']
    m2_vs_m3_ari = ari_matrix['Method2_ComBat']['Method3_AgeBinning']
    
    avg_ari = (m1_vs_m2_ari + m1_vs_m3_ari + m2_vs_m3_ari) / 3
    
    print(f"三种校正方法间Top-10代谢物平均Jaccard: {avg_jac:.3f}")
    print(f"三种校正方法间NMF亚型平均ARI: {avg_ari:.3f}")
    
    if avg_jac > 0.5 and avg_ari > 0.5:
        print(f"\n✅ 三种方法结果高度一致，说明结论对残差化方法选择不敏感")
        print(f"   线性回归残差化的选择是合理的")
    elif avg_jac > 0.3:
        print(f"\n⚠️ 三种方法结果中等一致，部分代谢物和亚型对方法选择敏感")
        print(f"   建议在讨论中提及方法选择的影响")
    else:
        print(f"\n🔴 三种方法结果差异较大，残差化方法选择对结论有显著影响")
        print(f"   需要更深入讨论方法选择的依据")
    
    # 校正 vs 未校正
    uncorrected_acc = results['Uncorrected']['accuracy']
    corrected_accs = [results[m]['accuracy'] for m in ['Method1_LinearRegression', 'Method2_ComBat', 'Method3_AgeBinning']]
    avg_corrected_acc = np.mean(corrected_accs)
    
    print(f"\n未校正准确率: {uncorrected_acc:.1%}")
    print(f"三种校正方法平均准确率: {avg_corrected_acc:.1%}")
    print(f"准确率下降: {(uncorrected_acc - avg_corrected_acc)*100:.1f} 个百分点")
    
    print(f"\n结果已保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
