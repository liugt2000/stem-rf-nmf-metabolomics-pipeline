# -*- coding: utf-8 -*-
"""
分析1：置换检验 + Bootstrap CI + LOOCV
解决"46.5%是否显著高于随机"的核心质疑

输出：
- LOOCV准确率和混淆矩阵
- 置换检验p值（1000次置换）
- Bootstrap 95% CI（1000次）
- 与原始10折CV结果对比
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import sys
import time
import json

sys.path.insert(0, r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析')
from analysis_utils import (load_expression_matrix, load_clinical_data, 
                            filter_by_level, log_transform, age_residualization,
                            get_sample_groups, GROUP_ORDER)

# ========== 配置 ==========
N_ESTIMATORS = 1000  # RF树数量（和论文一致）
N_PERMUTATIONS = 500  # 置换次数
N_BOOTSTRAP = 1000  # Bootstrap次数
RANDOM_STATE = 42
OUTPUT_DIR = r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析\Analysis_Results'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_rf_cv(X, y, cv_method='5fold', n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE):
    """
    运行RF交叉验证（不使用class_weight，和论文一致）
    
    参数:
        X: 特征矩阵 (样本 × 特征)
        y: 标签
        cv_method: '5fold' / 'loocv' / 'repeated10' (5次重复10折)
    
    返回:
        accuracy, y_true, y_pred, y_proba
    """
    if cv_method == '5fold':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    elif cv_method == 'loocv':
        cv = LeaveOneOut()
    elif cv_method == 'repeated10':
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=random_state)
    else:
        raise ValueError(f"Unknown cv_method: {cv_method}")
    
    y_true_all = []
    y_pred_all = []
    y_proba_all = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features='sqrt',
            # 不使用class_weight='balanced'，和论文一致
            random_state=random_state,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_proba_all.append(y_proba)
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_proba_all = np.vstack(y_proba_all)
    
    accuracy = accuracy_score(y_true_all, y_pred_all)
    
    return accuracy, y_true_all, y_pred_all, y_proba_all


def permutation_test(X, y, n_permutations=N_PERMUTATIONS, observed_acc=None):
    """
    置换检验：随机打乱标签，计算零分布
    
    返回:
        perm_accuracies: 置换后的准确率分布
        p_value: p值
    """
    if observed_acc is None:
        observed_acc, _, _, _ = run_rf_cv(X, y, '5fold')
    
    perm_accuracies = []
    rng = np.random.RandomState(RANDOM_STATE)
    
    print(f"  置换检验开始 ({n_permutations}次)...")
    start = time.time()
    
    for i in range(n_permutations):
        # 随机打乱标签
        y_perm = rng.permutation(y)
        
        # 5折CV
        acc, _, _, _ = run_rf_cv(X, y_perm, '5fold', random_state=RANDOM_STATE + i)
        perm_accuracies.append(acc)
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"    完成 {i+1}/{n_permutations} ({elapsed:.0f}s), "
                  f"当前均值={np.mean(perm_accuracies):.3f}")
    
    perm_accuracies = np.array(perm_accuracies)
    
    # p值：(置换准确率 >= 观测准确率 的次数 + 1) / (总次数 + 1)
    p_value = (np.sum(perm_accuracies >= observed_acc) + 1) / (n_permutations + 1)
    
    elapsed = time.time() - start
    print(f"  置换检验完成 ({elapsed:.0f}s), p={p_value:.4f}")
    
    return perm_accuracies, p_value


def bootstrap_ci_from_loocv(y_true, y_pred, n_bootstrap=N_BOOTSTRAP):
    """
    Bootstrap 95% CI — 标准方法：bootstrap LOOCV预测结果
    
    参数:
        y_true: LOOCV真实标签 (n_samples,)
        y_pred: LOOCV预测标签 (n_samples,)
        n_bootstrap: bootstrap次数
    
    返回:
        boot_accuracies: bootstrap准确率分布
        ci: (lower, upper) 95% CI
        per_class_ci: 每类灵敏度的CI
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n_samples = len(y_true)
    classes = GROUP_ORDER
    
    correct = (y_true == y_pred).astype(int)
    
    boot_accuracies = []
    boot_sensitivities = {c: [] for c in classes}
    
    print(f"  Bootstrap开始 ({n_bootstrap}次，基于LOOCV预测结果)...")
    start = time.time()
    
    for i in range(n_bootstrap):
        # 有放回抽样样本索引
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        
        # 计算bootstrap样本的准确率
        boot_acc = correct[indices].mean()
        boot_accuracies.append(boot_acc)
        
        # 每类灵敏度
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        cm = confusion_matrix(y_true_boot, y_pred_boot, labels=classes)
        for j, c in enumerate(classes):
            if cm[j].sum() > 0:
                sens = cm[j, j] / cm[j].sum()
            else:
                sens = np.nan
            boot_sensitivities[c].append(sens)
        
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            print(f"    完成 {i+1}/{n_bootstrap} ({elapsed:.0f}s)")
    
    boot_accuracies = np.array(boot_accuracies)
    
    # 95% CI
    ci_lower = np.percentile(boot_accuracies, 2.5)
    ci_upper = np.percentile(boot_accuracies, 97.5)
    
    # 每类灵敏度CI
    per_class_ci = {}
    for c in classes:
        sens = np.array(boot_sensitivities[c])
        sens = sens[~np.isnan(sens)]
        if len(sens) > 0:
            per_class_ci[c] = (np.percentile(sens, 2.5), np.percentile(sens, 97.5), np.mean(sens))
        else:
            per_class_ci[c] = (np.nan, np.nan, np.nan)
    
    elapsed = time.time() - start
    print(f"  Bootstrap完成 ({elapsed:.0f}s)")
    print(f"  准确率 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    return boot_accuracies, (ci_lower, ci_upper), per_class_ci


def main():
    print("=" * 70)
    print("分析1：置换检验 + Bootstrap CI + LOOCV")
    print("=" * 70)
    
    # ========== 1. 加载和预处理数据 ==========
    print("\n[1/5] 加载数据...")
    annot, expr = load_expression_matrix()
    crf = load_clinical_data()
    
    # 过滤B(i)级
    annot_bi, expr_bi = filter_by_level(annot, expr, 'B(i)')
    
    # log转换
    expr_log = log_transform(expr_bi)
    
    # 年龄残差化
    expr_resid = age_residualization(expr_log, crf)
    
    # 获取样本和标签
    samples, groups = get_sample_groups(expr_resid, crf)
    X = expr_resid[samples].values.T  # 样本 × 特征
    y = groups
    
    print(f"\n数据矩阵: X={X.shape}, y={y.shape}")
    print(f"分组分布: {dict(pd.Series(y).value_counts())}")
    
    # ========== 2. LOOCV ==========
    print("\n[2/5] LOOCV (留一法交叉验证)...")
    start = time.time()
    loocv_acc, y_true_loocv, y_pred_loocv, y_proba_loocv = run_rf_cv(X, y, 'loocv')
    elapsed = time.time() - start
    
    # 混淆矩阵
    cm_loocv = confusion_matrix(y_true_loocv, y_pred_loocv, labels=GROUP_ORDER)
    print(f"  LOOCV准确率: {loocv_acc:.4f} ({elapsed:.0f}s)")
    print(f"  混淆矩阵:\n{cm_loocv}")
    
    # 每类灵敏度
    print(f"  每类灵敏度:")
    for i, g in enumerate(GROUP_ORDER):
        if cm_loocv[i].sum() > 0:
            sens = cm_loocv[i, i] / cm_loocv[i].sum()
            print(f"    {g}: {sens:.1%} ({cm_loocv[i,i]}/{cm_loocv[i].sum()})")
    
    # ========== 3. 5次重复×10折CV（主准确率，和论文一致） ==========
    print("\n[3/5] 5次重复×10折CV（主准确率）...")
    start = time.time()
    rep10_acc, y_true_rep10, y_pred_rep10, y_proba_rep10 = run_rf_cv(X, y, 'repeated10')
    elapsed = time.time() - start
    cm_rep10 = confusion_matrix(y_true_rep10, y_pred_rep10, labels=GROUP_ORDER)
    print(f"  5次重复×10折CV准确率: {rep10_acc:.4f} ({elapsed:.0f}s)")
    
    # ========== 4. 置换检验（用5折CV加速） ==========
    print("\n[4/5] 置换检验...")
    perm_accs, p_value = permutation_test(X, y, n_permutations=N_PERMUTATIONS, 
                                            observed_acc=rep10_acc)
    
    # ========== 5. Bootstrap CI（基于LOOCV预测结果） ==========
    print("\n[5/5] Bootstrap 95% CI...")
    boot_accs, boot_ci, per_class_ci = bootstrap_ci_from_loocv(y_true_loocv, y_pred_loocv, 
                                                                   n_bootstrap=N_BOOTSTRAP)
    
    # ========== 保存结果 ==========
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    results = {
        'loocv_accuracy': float(loocv_acc),
        'loocv_confusion_matrix': cm_loocv.tolist(),
        'repeated10_accuracy': float(rep10_acc),
        'repeated10_confusion_matrix': cm_rep10.tolist(),
        'permutation_p_value': float(p_value),
        'permutation_mean': float(np.mean(perm_accs)),
        'permutation_std': float(np.std(perm_accs)),
        'bootstrap_ci_lower': float(boot_ci[0]),
        'bootstrap_ci_upper': float(boot_ci[1]),
        'bootstrap_mean': float(np.mean(boot_accs)),
        'chance_level': 0.25,
        'n_permutations': N_PERMUTATIONS,
        'n_bootstrap': N_BOOTSTRAP,
        'n_estimators': N_ESTIMATORS,
        'cv_method': '5 repeats x 10-fold (main), LOOCV, 5-fold (permutation)',
        'class_weight': 'None (not balanced)',
        'per_class_sensitivity_loocv': {},
        'per_class_sensitivity_bootstrap_ci': {},
    }
    
    for i, g in enumerate(GROUP_ORDER):
        if cm_loocv[i].sum() > 0:
            sens = cm_loocv[i, i] / cm_loocv[i].sum()
            results['per_class_sensitivity_loocv'][g] = float(sens)
        if g in per_class_ci:
            results['per_class_sensitivity_bootstrap_ci'][g] = {
                'lower': float(per_class_ci[g][0]),
                'upper': float(per_class_ci[g][1]),
                'mean': float(per_class_ci[g][2])
            }
    
    # 保存JSON
    with open(f'{OUTPUT_DIR}\\analysis1_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存numpy数组
    np.save(f'{OUTPUT_DIR}\\analysis1_perm_accuracies.npy', perm_accs)
    np.save(f'{OUTPUT_DIR}\\analysis1_boot_accuracies.npy', boot_accs)
    
    print(f"\nLOOCV准确率: {loocv_acc:.1%}")
    print(f"5次重复×10折CV准确率: {rep10_acc:.1%}")
    print(f"置换检验 p值: {p_value:.4f}")
    print(f"Bootstrap 95% CI: [{boot_ci[0]:.1%}, {boot_ci[1]:.1%}]")
    print(f"随机水平: 25%")
    print(f"\n结果已保存到: {OUTPUT_DIR}")
    
    # 关键结论
    print("\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)
    if p_value < 0.05:
        print(f"✅ 年龄残差化后的准确率 ({rep10_acc:.1%}) 显著高于随机水平 (25%)")
        print(f"   置换检验 p={p_value:.4f}")
    else:
        print(f"⚠️ 年龄残差化后的准确率 ({rep10_acc:.1%}) 未显著高于随机水平")
    
    if boot_ci[0] > 0.25:
        print(f"✅ Bootstrap 95% CI下限 ({boot_ci[0]:.1%}) 高于随机水平 (25%)")
    else:
        print(f"⚠️ Bootstrap 95% CI包含随机水平 (25%)")


if __name__ == '__main__':
    main()
