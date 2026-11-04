# -*- coding: utf-8 -*-
"""
验证核心数字（A1/A2/A4 实证裁决）
- A2: B(i) 特征数（1900 vs 1966）
- A1: 未校正/校正后 RF 的 pooled accuracy vs per-fold mean±std
       （验证 46.50%±14.72% 是否为 per-fold 口径，48.8% 是否为 pooled 口径）
- A4: 未校正/校正后的 HC(CG) 灵敏度（验证 100%/40% vs 真实值）
运行: D:\ANAconda\envs\cptac_env\python.exe verify_core_numbers.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
import json

sys.path.insert(0, r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析')
from analysis_utils import (load_expression_matrix, load_clinical_data, filter_by_level,
                            log_transform, age_residualization, get_sample_groups, GROUP_ORDER)

RANDOM_STATE = 42


def run_rf_repeated_cv(X, y, n_repeats=5, n_splits=10, n_estimators=1000):
    """5次重复×10折分层CV，输出 pooled accuracy 与 per-fold mean±std"""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    y_true_all, y_pred_all = [], []
    fold_accs = []
    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = RandomForestClassifier(n_estimators=n_estimators, max_features='sqrt',
                                     random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        fold_accs.append(accuracy_score(y_test, y_pred))
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    pooled = accuracy_score(y_true_all, y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=GROUP_ORDER)
    fold_accs = np.array(fold_accs)
    return pooled, float(fold_accs.mean()), float(fold_accs.std()), cm


def sens(cm):
    return {g: float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0.0
            for i, g in enumerate(GROUP_ORDER)}


def main():
    out = {}
    annot, expr = load_expression_matrix()
    crf = load_clinical_data()
    annot_bi, expr_bi = filter_by_level(annot, expr, 'B(i)')
    out['n_bi_features'] = int(expr_bi.shape[0])
    expr_log = log_transform(expr_bi)

    # 未校正（B(i)级，无年龄残差化）
    samples, groups = get_sample_groups(expr_log, crf)
    X_unc = expr_log[samples].values.T
    pooled_u, mean_u, std_u, cm_u = run_rf_repeated_cv(X_unc, groups)
    out['uncorrected'] = {
        'pooled_accuracy': float(pooled_u),
        'per_fold_mean': mean_u,
        'per_fold_std': std_u,
        'sensitivity': sens(cm_u),
        'cm': cm_u.tolist(),
    }

    # 校正后（B(i)级，年龄残差化）
    expr_resid = age_residualization(expr_log, crf)
    samples_c, groups_c = get_sample_groups(expr_resid, crf)
    X_c = expr_resid[samples_c].values.T
    pooled_c, mean_c, std_c, cm_c = run_rf_repeated_cv(X_c, groups_c)
    out['corrected'] = {
        'pooled_accuracy': float(pooled_c),
        'per_fold_mean': mean_c,
        'per_fold_std': std_c,
        'sensitivity': sens(cm_c),
        'cm': cm_c.tolist(),
    }

    # 与稿件数字对照
    out['comparison'] = {
        'paper_uncorrected_baseline': '92.31%',
        'paper_corrected_main': '46.50% ± 14.72%',
        'paper_rep10_new_analysis': '48.8%',
        'paper_counterfactual_corrected': '47.6%',
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
