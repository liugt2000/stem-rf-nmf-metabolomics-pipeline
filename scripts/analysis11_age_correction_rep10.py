# -*- coding: utf-8 -*-
"""分析11：三种年龄校正方法在统一 5x10 折口径下的重算（供 3.5.2 节替换 5 折数字）。"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score)
from sklearn.model_selection import RepeatedStratifiedKFold

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE = os.environ.get('METABO_DATA_DIR', str(ROOT / 'data'))
sys.path.insert(0, str(SCRIPT_DIR))
from analysis_utils import (GROUP_ORDER, age_binning_standardize, age_residualization,
                            combat_adjust, filter_by_level, get_sample_groups,
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


def cv_eval(X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=RANDOM_STATE)
    fold, yt, yp = [], [], []
    for tr, te in cv.split(X, y):
        clf = RandomForestClassifier(n_estimators=N_TREES, max_features='sqrt',
                                     random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        fold.append(accuracy_score(y[te], p))
        yt.extend(y[te])
        yp.extend(p)
    yt, yp = np.array(yt), np.array(yp)
    cm = confusion_matrix(yt, yp, labels=GROUP_ORDER)
    return {
        'fold_acc_mean': round(float(np.mean(fold)), 4),
        'fold_acc_sd': round(float(np.std(fold)), 4),
        'pooled': round(float(accuracy_score(yt, yp)), 4),
        'kappa': round(float(cohen_kappa_score(yt, yp)), 3),
        'macro_f1': round(float(f1_score(yt, yp, average='macro')), 3),
        'per_class': {g: round(float(cm[i, i] / cm[i].sum()), 3) for i, g in enumerate(GROUP_ORDER)},
    }


def main():
    annot, expr = load_expression_matrix()
    crf = load_crf()
    annot_bi, expr_bi = filter_by_level(annot, expr, 'B(i)')
    expr_log = log_transform(expr_bi)

    methods = {
        'Uncorrected': expr_log.copy(),
        'LinearRegression': age_residualization(expr_log, crf,
                                                covariates=['Age', 'Education_code', 'Gender_code']),
        'ComBat': combat_adjust(expr_log, crf, batch_var='Age'),
        'AgeBinning': age_binning_standardize(expr_log, crf, n_bins=3),
    }

    out = {}
    for name, dat in methods.items():
        t0 = time.time()
        samples, groups = get_sample_groups(dat, crf)
        X = dat[samples].values.T
        y = np.asarray(groups)
        r = cv_eval(X, y)
        r['n_features'] = int(X.shape[1])
        out[name] = r
        print(f"{name:18s} acc={r['fold_acc_mean']:.4f}±{r['fold_acc_sd']:.4f} "
              f"pooled={r['pooled']:.4f} kappa={r['kappa']:.3f} AD={r['per_class']['AD']:.3f} "
              f"({time.time()-t0:.0f}s)")

    with open(os.path.join(OUT_DIR, 'age_correction_rep10.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    tab = pd.DataFrame([{
        'Method': k,
        'Accuracy': f"{v['fold_acc_mean']*100:.1f}% ± {v['fold_acc_sd']*100:.1f}%",
        'Pooled': f"{v['pooled']*100:.1f}%",
        'Kappa': v['kappa'],
        'HC': f"{v['per_class']['CG']*100:.0f}%",
        'SCD': f"{v['per_class']['SCD']*100:.0f}%",
        'MCI': f"{v['per_class']['MCI']*100:.0f}%",
        'AD': f"{v['per_class']['AD']*100:.0f}%",
    } for k, v in out.items()])
    tab.to_csv(os.path.join(OUT_DIR, 'age_correction_rep10.csv'), index=False, encoding='utf-8-sig')
    print()
    print(tab.to_string(index=False))


if __name__ == '__main__':
    main()
