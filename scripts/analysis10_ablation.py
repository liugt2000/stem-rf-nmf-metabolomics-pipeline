# -*- coding: utf-8 -*-
"""
分析10：Pipeline 消融实验（ablation study）

在同一批 84 例、同一套 CV（5 次重复 × 10 折，1000 树）下，逐个打开/关闭 pipeline 的两个
可分离组件：
  1) MSI 等级过滤（只保留 B(i) vs 纳入 B(ii)）
  2) 人口学残差化（年龄/教育/性别 vs 不校正）

输出阶梯表：总体准确率（均值±SD）、κ、AD 敏感度、每类灵敏度、top-50 中 B(ii) 占比、
top-10 重要性跨重复 Jaccard、NMF 亚型与年龄的均值差及其 P 值。

用法：
  python analysis10_ablation.py --quick   # 小规模冒烟测试
  python analysis10_ablation.py           # 全量
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE = os.environ.get('METABO_DATA_DIR', str(ROOT / 'data'))
sys.path.insert(0, str(SCRIPT_DIR))
from analysis_utils import (GROUP_ORDER, age_residualization, filter_by_level,
                            get_sample_groups, load_expression_matrix, log_transform)

OUT_DIR = os.environ.get('METABO_OUT_DIR', str(ROOT / 'results'))
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42
GROUP_STAGE = {'CG': 0, 'SCD': 1, 'MCI': 2, 'AD': 3}


def load_crf():
    """从转换好的 crf.csv 读取临床表（避免 xlrd 依赖），复刻原编码口径。"""
    crf = pd.read_csv(os.path.join(BASE, 'crf.csv'), encoding='utf-8-sig')
    crf = crf.set_index('Sample Analysis Name')
    crf['Group'] = crf['Group'].astype(str).str.strip()
    crf['Gender_code'] = (crf['Gender'] == 'Male').astype(int)
    edu_map = {'Primary School': 1, 'Middle School': 2, 'High School': 3,
               'College': 4, 'Bachelor': 4, 'Master': 5, 'PhD': 6}
    crf['Education_code'] = crf['Education'].map(edu_map).fillna(3)
    crf['Stage'] = crf['Group'].map(GROUP_STAGE)
    print(f"临床数据: {crf.shape[0]} 样本; 分组 {dict(crf['Group'].value_counts())}; "
          f"年龄 {crf['Age'].mean():.1f} ± {crf['Age'].std():.1f}")
    return crf


def build_matrix(annot, expr, crf, level_filter, residualize):
    """返回 (X, y, samples, n_features)。level_filter: 'B(i)' 或 'B(i)+B(ii)'。"""
    if level_filter == 'B(i)':
        a, e = filter_by_level(annot, expr, 'B(i)')
    else:
        mask = annot['level'].isin(['B(i)', 'B(ii)'])
        a = annot[mask].copy()
        e = expr.loc[a['metab_id'].values].copy()
    e_log = log_transform(e)
    if residualize:
        e_use = age_residualization(e_log, crf)
    else:
        e_use = e_log
    samples, groups = get_sample_groups(e_use, crf)
    X = e_use[samples].values.T
    return X, np.asarray(groups), samples, a


def rf_cv(X, y, n_estimators, cv, want_importance_repeats=False):
    """跑 CV，返回折准确率列表、汇总预测、重要性、每重复的 top10 集合。"""
    fold_acc, y_true_all, y_pred_all = [], [], []
    imp_acc = np.zeros(X.shape[1])
    n_fits = 0
    top10_sets = []

    for train_idx, test_idx in cv.split(X, y):
        clf = RandomForestClassifier(n_estimators=n_estimators, max_features='sqrt',
                                     random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        fold_acc.append(accuracy_score(y[test_idx], pred))
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(pred)
        imp_acc += clf.feature_importances_
        n_fits += 1
        if want_importance_repeats and n_fits % 10 == 0:
            top10_sets.append(tuple(np.argsort(imp_acc)[-10:][::-1]))

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=GROUP_ORDER)
    per_class = {g: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else np.nan
                 for i, g in enumerate(GROUP_ORDER)}

    jac = np.nan
    if len(top10_sets) >= 2:
        vals = []
        for i in range(len(top10_sets)):
            for j in range(i + 1, len(top10_sets)):
                a, b = set(top10_sets[i]), set(top10_sets[j])
                vals.append(len(a & b) / len(a | b))
        jac = float(np.mean(vals))

    return {
        'fold_acc_mean': float(np.mean(fold_acc)),
        'fold_acc_sd': float(np.std(fold_acc)),
        'accuracy_pooled': float(accuracy_score(y_true_all, y_pred_all)),
        'kappa': float(cohen_kappa_score(y_true_all, y_pred_all)),
        'macro_f1': float(f1_score(y_true_all, y_pred_all, average='macro')),
        'per_class_sensitivity': per_class,
        'confusion_matrix': cm.tolist(),
        'importance': (imp_acc / max(n_fits, 1)),
        'top10_jaccard': jac,
        'n_folds': len(fold_acc),
    }


def nmf_check(X, samples, crf):
    """k=2 NMF，返回亚型与年龄/疾病的关联。"""
    Xn = X - X.min() + 1e-10
    model = NMF(n_components=2, init='nndsvda', max_iter=1000, random_state=RANDOM_STATE)
    W = model.fit_transform(Xn)
    st = np.argmax(W, axis=1)
    d = crf.loc[samples].copy()
    d['subtype'] = st
    a0 = d[d['subtype'] == 0]['Age'].values
    a1 = d[d['subtype'] == 1]['Age'].values
    p_age = np.nan
    diff_age = np.nan
    if len(a0) > 1 and len(a1) > 1:
        _, p_age = stats.ttest_ind(a0, a1)
        diff_age = float(np.mean(a0) - np.mean(a1))
    ct = pd.crosstab(d['subtype'], d['Group'])
    try:
        _, p_dis, _, _ = stats.chi2_contingency(ct)
    except Exception:
        p_dis = np.nan
    return {
        'subtype_sizes': [int((st == 0).sum()), int((st == 1).sum())],
        'age_mean_diff_years': None if np.isnan(diff_age) else round(diff_age, 2),
        'age_p_value': None if np.isnan(p_age) else float(p_age),
        'disease_chi2_p': None if np.isnan(p_dis) else float(p_dis),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--trees', type=int, default=1000)
    args = ap.parse_args()

    n_trees = 100 if args.quick else args.trees
    print(f"[config] trees={n_trees} quick={args.quick}")

    annot, expr = load_expression_matrix()
    crf = load_crf()

    configs = [
        ('full',        'B(i)',      True),
        ('no_resid',    'B(i)',      False),
        ('bi_ii_resid', 'B(i)+B(ii)', True),
        ('bi_ii_raw',   'B(i)+B(ii)', False),
    ]

    out = {'meta': {'n_trees': n_trees, 'cv': '5x10 RepeatedStratifiedKFold',
                    'quick': args.quick}, 'configs': {}}

    for name, lvl, resid in configs:
        t0 = time.time()
        print(f"\n=== {name}: level={lvl}, residualize={resid}")
        X, y, samples, a = build_matrix(annot, expr, crf, lvl, resid)
        print(f"    X={X.shape}, classes={dict(pd.Series(y).value_counts())}")

        cv_rep = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=RANDOM_STATE)
        r = rf_cv(X, y, n_trees, cv_rep, want_importance_repeats=True)
        imp = r.pop('importance')
        top_idx = np.argsort(imp)[-50:][::-1]
        lv = a.iloc[top_idx]['level'].values
        top50 = {'B(ii)_n': int((lv == 'B(ii)').sum()),
                 'B(ii)_pct': float((lv == 'B(ii)').mean() * 100)}
        r['top50'] = top50
        r['nmf'] = nmf_check(X, samples, crf)

        cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        r5 = rf_cv(X, y, n_trees, cv5)
        r['five_fold_accuracy'] = r5['accuracy_pooled']

        out['configs'][name] = {
            'level_filter': lvl, 'residualize': resid, 'n_features': int(X.shape[1]),
            **{k: v for k, v in r.items() if k != 'confusion_matrix'},
            'confusion_matrix': r['confusion_matrix'],
        }
        c = out['configs'][name]
        print(f"    acc={c['fold_acc_mean']:.4f}±{c['fold_acc_sd']:.4f} "
              f"pooled={c['accuracy_pooled']:.4f} kappa={c['kappa']:.3f} "
              f"AD_sens={c['per_class_sensitivity']['AD']:.3f} "
              f"top50_Bii={c['top50']['B(ii)_pct']:.1f}% "
              f"5fold={c['five_fold_accuracy']:.4f} "
              f"nmf_age_diff={c['nmf']['age_mean_diff_years']} "
              f"({time.time()-t0:.0f}s)")

    with open(os.path.join(OUT_DIR, 'ablation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    rows = []
    for k, c in out['configs'].items():
        rows.append({
            'Configuration': k,
            'Level filter': c['level_filter'],
            'Residualized': 'yes' if c['residualize'] else 'no',
            'N features': c['n_features'],
            'Accuracy (5x10 CV)': f"{c['fold_acc_mean']*100:.1f}% ± {c['fold_acc_sd']*100:.1f}%",
            'Pooled accuracy': f"{c['accuracy_pooled']*100:.1f}%",
            'Kappa': round(c['kappa'], 3),
            'Macro F1': round(c['macro_f1'], 3),
            'AD sensitivity': f"{c['per_class_sensitivity']['AD']*100:.0f}%",
            'SCD sensitivity': f"{c['per_class_sensitivity']['SCD']*100:.0f}%",
            'MCI sensitivity': f"{c['per_class_sensitivity']['MCI']*100:.0f}%",
            'HC sensitivity': f"{c['per_class_sensitivity']['CG']*100:.0f}%",
            'Top-50 B(ii) %': round(c['top50']['B(ii)_pct'], 1),
            'Top-10 Jaccard': None if c['top10_jaccard'] != c['top10_jaccard'] else round(c['top10_jaccard'], 3),
            'NMF age diff (y)': c['nmf']['age_mean_diff_years'],
            'NMF age p': None if c['nmf']['age_p_value'] is None else round(c['nmf']['age_p_value'], 4),
            '5-fold accuracy': f"{c['five_fold_accuracy']*100:.1f}%",
        })
    tab = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'ablation_ladder.csv')
    tab.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print("\n" + tab.to_string(index=False))
    print("\nSaved:", csv_path)


if __name__ == '__main__':
    main()
