"""
分析4：多分类器比较（简化版）
比较5种分类器在年龄残差化数据上的性能
"""
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

sys.path.insert(0, r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析')
from analysis_utils import load_expression_matrix, load_clinical_data, filter_by_level, log_transform, age_residualization, get_sample_groups, GROUP_ORDER

OUTPUT_DIR = r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析\Analysis_Results'
RANDOM_STATE = 42

print("=" * 70)
print("分析4：多分类器比较")
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

# ========== 2. 定义分类器 ==========
print("\n[2/3] 定义分类器...")

def make_classifiers():
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=1000, max_features='sqrt',
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=RANDOM_STATE, subsample=0.8
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0, penalty='l2', max_iter=5000,
            random_state=RANDOM_STATE
        ),
        'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    }

# 需要标准化的分类器
NEED_SCALING = ['SVM (RBF)', 'Logistic Regression', 'k-NN (k=5)']

for name in make_classifiers():
    print(f"  - {name}")

# ========== 3. 运行CV ==========
print("\n[3/3] 运行交叉验证（5次重复×10折）...")

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=RANDOM_STATE)
scaler = StandardScaler()

results = {}

for clf_name in make_classifiers():
    print(f"\n  --- {clf_name} ---")
    clfs = make_classifiers()
    clf = clfs[clf_name]
    
    y_true_all = []
    y_pred_all = []
    fold_accuracies = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 标准化（如果需要）
        if clf_name in NEED_SCALING:
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
        else:
            X_train_s = X_train
            X_test_s = X_test
        
        # 克隆分类器并训练
        clf_fold = clone(clf)
        clf_fold.fit(X_train_s, y_train)
        y_pred = clf_fold.predict(X_test_s)
        
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    overall_acc = accuracy_score(y_true_all, y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=GROUP_ORDER)
    macro_f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    
    sensitivities = {}
    for i, g in enumerate(GROUP_ORDER):
        if cm[i].sum() > 0:
            sensitivities[g] = float(cm[i, i] / cm[i].sum())
    
    results[clf_name] = {
        'accuracy': float(overall_acc),
        'accuracy_std': float(np.std(fold_accuracies)),
        'macro_f1': float(macro_f1),
        'confusion_matrix': cm.tolist(),
        'sensitivities': sensitivities,
    }
    
    print(f"    准确率: {overall_acc:.1%} (±{np.std(fold_accuracies):.1%})")
    print(f"    Macro F1: {macro_f1:.3f}")
    for g in GROUP_ORDER:
        print(f"    {g}灵敏度: {sensitivities.get(g, 0):.0%}")

# ========== 4. 汇总 ==========
print("\n" + "=" * 70)
print("结果汇总")
print("=" * 70)

comparison_table = []
for clf_name, res in results.items():
    row = {
        'Classifier': clf_name,
        'Accuracy': f"{res['accuracy']:.1%}",
        'Std': f"{res['accuracy_std']:.1%}",
        'Macro_F1': f"{res['macro_f1']:.3f}",
        'CG': f"{res['sensitivities'].get('CG', 0):.0%}",
        'SCD': f"{res['sensitivities'].get('SCD', 0):.0%}",
        'MCI': f"{res['sensitivities'].get('MCI', 0):.0%}",
        'AD': f"{res['sensitivities'].get('AD', 0):.0%}",
    }
    comparison_table.append(row)

comparison_df = pd.DataFrame(comparison_table)
print("\n分类器性能对比:")
print(comparison_df.to_string(index=False))

best_clf = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n最优分类器: {best_clf[0]} (准确率: {best_clf[1]['accuracy']:.1%})")

# ========== 保存 ==========
output = {
    'classifiers': results,
    'comparison_table': comparison_table,
    'best_classifier': best_clf[0],
    'best_accuracy': best_clf[1]['accuracy'],
    'cv_method': '5 repeats x 10-fold stratified CV',
}

with open(f'{OUTPUT_DIR}\\analysis4_classifier_comparison.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

comparison_df.to_csv(f'{OUTPUT_DIR}\\analysis4_classifier_comparison.csv',
                      index=False, encoding='utf-8-sig')

print(f"\n结果已保存到: {OUTPUT_DIR}")
