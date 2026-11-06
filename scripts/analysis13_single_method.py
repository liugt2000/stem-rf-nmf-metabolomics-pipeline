# -*- coding: utf-8 -*-
"""分析13：3.8 节"与单方法对比"所需的单方法基线。
在同一 B(i) 残差化矩阵上，比较：
  - 全特征 RF（= 集成管线内的分类器）
  - ANOVA 预选 top-50 / top-200 特征 + RF（文献常见做法，折内选择避免泄漏）
输出 repeated10 口径准确率。"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
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
    crf = pd.read_csv(os.path.join(BASE, 'crf.csv'), encoding='utf-8-sig')
    crf = crf.set_index('Sample Analysis Name')
    crf['Group'] = crf['Group'].astype(str).str.strip()
    crf['Gender_code'] = (crf['Gender'] == 'Male').astype(int)
    edu_map = {'Primary School': 1, 'Middle School': 2, 'High School': 3,
               'College': 4, 'Bachelor': 4, 'Master': 5, 'PhD': 6}
    crf['Education_code'] = crf['Education'].map(edu_map).fillna(3)
    crf['Stage'] = crf['Group'].map(GROUP_STAGE)
    return crf


def main():
    annot, expr = load_expression_matrix()
    crf = load_crf()
    annot_bi, expr_bi = filter_by_level(annot, expr, 'B(i)')
    expr_res = age_residualization(log_transform(expr_bi), crf,
                                   covariates=['Age', 'Education_code', 'Gender_code'])
    samples, groups = get_sample_groups(expr_res, crf)
    X = expr_res[samples].values.T
    y = np.asarray(groups)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=RANDOM_STATE)
    out = {}
    for label, k in [('all_features', None), ('anova_top50', 50), ('anova_top200', 200)]:
        t0 = time.time()
        fold = []
        yt, yp = [], []
        for tr, te in cv.split(X, y):
            steps = []
            if k is not None:
                steps.append(('select', SelectKBest(f_classif, k=k)))
            steps.append(('rf', RandomForestClassifier(n_estimators=N_TREES, max_features='sqrt',
                                                       random_state=RANDOM_STATE, n_jobs=-1)))
            pipe = Pipeline(steps)
            pipe.fit(X[tr], y[tr])
            p = pipe.predict(X[te])
            fold.append(accuracy_score(y[te], p))
            yt.extend(y[te])
            yp.extend(p)
        out[label] = {
            'n_features_used': k if k else int(X.shape[1]),
            'fold_acc_mean': round(float(np.mean(fold)), 4),
            'fold_acc_sd': round(float(np.std(fold)), 4),
            'pooled': round(float(accuracy_score(np.array(yt), np.array(yp))), 4),
            'seconds': round(time.time() - t0, 1),
        }
        print(f"{label:14s} k={out[label]['n_features_used']:5d} "
              f"acc={out[label]['fold_acc_mean']:.4f}±{out[label]['fold_acc_sd']:.4f} "
              f"pooled={out[label]['pooled']:.4f} ({out[label]['seconds']:.0f}s)")

    with open(os.path.join(OUT_DIR, 'single_method_baseline.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
