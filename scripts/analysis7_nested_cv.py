"""
分析7：嵌套交叉验证（Nested CV）
外层CV：评估泛化性能（5次重复×5折）
内层CV：特征选择（ANOVA top-k）+ RF超参数优化
"""
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析')
from analysis_utils import load_expression_matrix, load_clinical_data, filter_by_level, log_transform, age_residualization, get_sample_groups, GROUP_ORDER

OUTPUT_DIR = r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析\Analysis_Results'
RANDOM_STATE = 42

print("=" * 70)
print("分析7：嵌套交叉验证（Nested CV）")
print("=" * 70)

# ========== 1. 加载数据 ==========
print("\n[1/3] 加载数据...")
annot, expr = load_expression_matrix()
crf = load_clinical_data()
annot_bi, expr_bi = filter_by_level(annot, expr, 'B(i)')
expr_log = log_transform(expr_bi)
expr_resid = age_residualization(expr_log, crf)
samples, groups = get_sample_groups(expr_resid, crf)
X = expr_resid[samples].values.T
y = np.array(groups)
print(f"  矩阵: {X.shape}")

# ========== 2. 嵌套交叉验证 ==========
print("\n[2/3] 运行嵌套交叉验证...")

# 外层CV：5折，重复3次
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 内层参数网格
param_grid = {
    'rf__n_estimators': [500, 1000],
    'rf__max_depth': [None, 10, 20],
    'feature_selection__k': [100, 300, 500],
}

# Pipeline：特征选择 + RF
pipeline = Pipeline([
    ('feature_selection', SelectKBest(f_classif)),
    ('rf', RandomForestClassifier(max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)),
])

outer_accuracies = []
outer_f1 = []
y_true_all = []
y_pred_all = []
best_params_list = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n  外层折 {fold_idx+1}/5...")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 内层CV：网格搜索
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=inner_cv, scoring='accuracy',
        n_jobs=-1, refit=True, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_params_list.append(best_params)
    print(f"    最优参数: {best_params}")
    print(f"    内层CV最优准确率: {grid_search.best_score_:.3f}")
    
    # 在外层测试集上评估
    y_pred = grid_search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    outer_accuracies.append(acc)
    outer_f1.append(f1)
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)
    
    print(f"    外层测试准确率: {acc:.3f}, Macro F1: {f1:.3f}")

# ========== 3. 汇总结果 ==========
print("\n[3/3] 汇总结果...")

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)
overall_acc = accuracy_score(y_true_all, y_pred_all)
cm = confusion_matrix(y_true_all, y_pred_all, labels=GROUP_ORDER)
macro_f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)

# 每类灵敏度
sensitivities = {}
for i, g in enumerate(GROUP_ORDER):
    if cm[i].sum() > 0:
        sensitivities[g] = float(cm[i, i] / cm[i].sum())

# 最优参数统计
param_counts = {}
for params in best_params_list:
    for key, val in params.items():
        if key not in param_counts:
            param_counts[key] = {}
        param_counts[key][str(val)] = param_counts[key].get(str(val), 0) + 1

print(f"\n  嵌套CV总体准确率: {overall_acc:.4f} (±{np.std(outer_accuracies):.4f})")
print(f"  各折准确率: {[f'{a:.3f}' for a in outer_accuracies]}")
print(f"  Macro F1: {macro_f1:.4f}")
print(f"\n  每类灵敏度:")
for g in GROUP_ORDER:
    print(f"    {g}: {sensitivities.get(g, 0):.1%}")

print(f"\n  最优参数统计:")
for key, counts in param_counts.items():
    print(f"    {key}: {counts}")

# 与普通CV比较（之前的结果：48.8%）
print(f"\n  与普通5×10折CV比较:")
print(f"    普通CV准确率: 48.8%")
print(f"    嵌套CV准确率: {overall_acc:.1%}")
print(f"    差异: {(overall_acc - 0.488)*100:.1f}个百分点")

# ========== 保存结果 ==========
output = {
    'nested_cv_accuracy': float(overall_acc),
    'nested_cv_accuracy_std': float(np.std(outer_accuracies)),
    'fold_accuracies': [float(a) for a in outer_accuracies],
    'macro_f1': float(macro_f1),
    'confusion_matrix': cm.tolist(),
    'sensitivities': sensitivities,
    'best_params_per_fold': [{k: str(v) for k, v in p.items()} for p in best_params_list],
    'param_counts': param_counts,
    'comparison_with_regular_cv': {
        'regular_cv_accuracy': 0.488,
        'nested_cv_accuracy': float(overall_acc),
        'difference_pp': float((overall_acc - 0.488) * 100),
    },
    'outer_cv': '5-fold stratified',
    'inner_cv': '3-fold stratified with grid search',
}

with open(f'{OUTPUT_DIR}\\analysis7_nested_cv.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存到: {OUTPUT_DIR}")
