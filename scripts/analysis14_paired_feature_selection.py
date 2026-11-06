# -*- coding: utf-8 -*-
"""分析14：全特征 vs ANOVA 预选 top-50 的配对比较（同一 CV 划分逐折配对）。"""

import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE = os.environ.get('METABO_DATA_DIR', str(ROOT / 'data'))
sys.path.insert(0, str(SCRIPT_DIR))
from analysis_utils import (age_residualization, filter_by_level, get_sample_groups,
                            load_expression_matrix, log_transform)

OUT_DIR = os.environ.get('METABO_OUT_DIR', str(ROOT / 'results'))
RANDOM_STATE = 42
N_TREES = 1000
GROUP_STAGE = {'CG': 0, 'SCD': 1, 'MCI': 2, 'AD': 3}


def load_crf():
    crf = __import__('pandas').read_csv(os.path.join(BASE, 'crf.csv'), encoding='utf-8-sig')
    crf = crf.set_index('Sample Analysis Name')
    crf['Group'] = crf['Group'].astype(str).str.strip()
    crf['Gender_code'] = (crf['Gender'] == 'Male').astype(int)
    edu_map = {'Primary School': 1, 'Middle School': 2, 'High School': 3,
               'College': 4, 'Bachelor': 4, 'Master': 5, 'PhD': 6}
    crf['Education_code'] = crf['Education'].map(edu_map).fillna(3)
    crf['Stage'] = crf['Group'].map(GROUP_STAGE)
    return crf


def run(X, y, cv, k):
    fold, yt, yp = [], [], []
    for tr, te in cv.split(X, y):
        steps = []
        if k:
            steps.append(('select', SelectKBest(f_classif, k=k)))
        steps.append(('rf', RandomForestClassifier(n_estimators=N_TREES, max_features='sqrt',
                                                   random_state=RANDOM_STATE, n_jobs=-1)))
        pipe = Pipeline(steps)
        pipe.fit(X[tr], y[tr])
        p = pipe.predict(X[te])
        fold.append(accuracy_score(y[te], p))
        yt.extend(y[te])
        yp.extend(p)
    return np.array(fold), accuracy_score(np.array(yt), np.array(yp))


def main():
    import pandas as pd
    annot, expr = load_expression_matrix()
    crf = load_crf()
    annot_bi, expr_bi = filter_by_level(annot, expr, 'B(i)')
    expr_res = age_residualization(log_transform(expr_bi), crf,
                                   covariates=['Age', 'Education_code', 'Gender_code'])
    samples, groups = get_sample_groups(expr_res, crf)
    X = expr_res[samples].values.T
    y = np.asarray(groups)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=RANDOM_STATE)

    fa, pa = run(X, y, cv, None)
    fb, pb = run(X, y, cv, 50)

    d = fb - fa
    t_stat, p_t = stats.ttest_rel(fb, fa)
    w_stat, p_w = stats.wilcoxon(fb, fa)
    res = {
        'all_features': {'fold_mean': round(float(fa.mean()), 4), 'fold_sd': round(float(fa.std()), 4),
                         'pooled': round(float(pa), 4)},
        'anova_top50': {'fold_mean': round(float(fb.mean()), 4), 'fold_sd': round(float(fb.std()), 4),
                        'pooled': round(float(pb), 4)},
        'paired': {'mean_diff_pp': round(float(d.mean() * 100), 2),
                   'sd_diff_pp': round(float(d.std() * 100), 2),
                   't': round(float(t_stat), 3), 'p_ttest': round(float(p_t), 5),
                   'p_wilcoxon': round(float(p_w), 5),
                   'n_folds_top50_better': int((d > 0).sum()), 'n_folds': int(len(d))},
    }
    with open(os.path.join(OUT_DIR, 'single_method_paired.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
