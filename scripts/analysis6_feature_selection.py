"""
分析6：特征选择策略比较
比较4种特征选择策略对RF分类性能的影响：
1. 全部1966特征（当前）
2. 单变量筛选（ANOVA p<0.05）
3. 递归特征消除（RFE）
4. LASSO特征选择
"""
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

sys.path.insert(0, r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析')
from analysis_utils import load_expression_matrix, load_clinical_data, filter_by_level, log_transform, age_residualization, get_sample_groups, GROUP_ORDER

OUTPUT_DIR = r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析\Analysis_Results'
RANDOM_STATE = 42
N_ESTIMATORS = 1000

print("=" * 70)
print("分析6：特征选择策略比较")
print("=" * 70)

# ========== 1. 加载数据 ==========
print("\n[1/4] 加载数据...")
annot, expr = load_expression_matrix()
crf = load_clinical_data()
annot_bi, expr_bi = filter_by_level(annot, expr, 'B(i)')
expr_log = log_transform(expr_bi)
expr_resid = age_residualization(expr_log, crf)
samples, groups = get_sample_groups(expr_resid, crf)
X = expr_resid[samples].values.T
y = np.array(groups)
metabolite_ids = expr_bi.index.tolist()
metabolite_names = annot_bi['Metabolite'].values
print(f"  矩阵: {X.shape}")

# ========== 2. 定义特征选择策略 ==========
print("\n[2/4] 定义特征选择策略...")

# 策略1：全部特征（不做选择）
# 策略2：单变量ANOVA筛选（p<0.05）
selector_univariate = SelectKBest(f_classif, k=500)  # 先选top500，后面用p值过滤

# 策略3：RFE（递归特征消除，目标500个特征）
rf_for_rfe = RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                      random_state=RANDOM_STATE, n_jobs=-1)
selector_rfe = RFE(rf_for_rfe, n_features_to_select=200, step=0.1)

# 策略4：LASSO（Logistic Regression with L1）
lasso = LogisticRegression(penalty='l1', C=0.1, solver='saga',
                            max_iter=5000,
                            random_state=RANDOM_STATE)

strategies = {
    'All features (1966)': 'all',
    'Univariate ANOVA (top 500)': 'univariate',
    'RFE (top 200)': 'rfe',
    'LASSO (non-zero coef)': 'lasso',
}

for name in strategies:
    print(f"  - {name}")

# ========== 3. 运行CV ==========
print("\n[3/4] 运行交叉验证（5次重复×10折）...")

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=RANDOM_STATE)
scaler = StandardScaler()

results = {}
selected_features = {}

for strategy_name, strategy_type in strategies.items():
    print(f"\n  --- {strategy_name} ---")
    
    y_true_all = []
    y_pred_all = []
    fold_accuracies = []
    n_features_list = []
    feature_indices_all = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 特征选择
        if strategy_type == 'all':
            X_train_sel = X_train
            X_test_sel = X_test
            selected_idx = np.arange(X.shape[1])
        elif strategy_type == 'univariate':
            sel = SelectKBest(f_classif, k=500)
            X_train_sel = sel.fit_transform(X_train, y_train)
            X_test_sel = sel.transform(X_test)
            selected_idx = np.where(sel.get_support())[0]
        elif strategy_type == 'rfe':
            rf_rfe = RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                              random_state=RANDOM_STATE, n_jobs=-1)
            sel = RFE(rf_rfe, n_features_to_select=200, step=0.1)
            X_train_sel = sel.fit_transform(X_train, y_train)
            X_test_sel = sel.transform(X_test)
            selected_idx = np.where(sel.get_support())[0]
        elif strategy_type == 'lasso':
            # LASSO需要标准化
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            lasso_fold = LogisticRegression(penalty='l1', C=0.1, solver='saga',
                                              max_iter=5000,
                                              random_state=RANDOM_STATE)
            lasso_fold.fit(X_train_scaled, y_train)
            # 选择系数非零的特征（任何一类系数非零）
            coef_sum = np.sum(np.abs(lasso_fold.coef_), axis=0)
            selected_idx = np.where(coef_sum > 0)[0]
            if len(selected_idx) == 0:
                selected_idx = np.argsort(coef_sum)[-100:]  # 至少选100个
            X_train_sel = X_train[:, selected_idx]
            X_test_sel = X_test[:, selected_idx]
        
        n_features_list.append(len(selected_idx))
        feature_indices_all.append(selected_idx)
        
        # RF分类
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features='sqrt',
                                      random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train_sel, y_train)
        y_pred = rf.predict(X_test_sel)
        
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
    
    # 计算跨fold的特征选择稳定性（Jaccard指数）
    if len(feature_indices_all) > 1:
        jaccards = []
        for i in range(len(feature_indices_all)):
            for j in range(i+1, len(feature_indices_all)):
                set1 = set(feature_indices_all[i])
                set2 = set(feature_indices_all[j])
                if len(set1 | set2) > 0:
                    jaccards.append(len(set1 & set2) / len(set1 | set2))
        avg_jaccard = np.mean(jaccards) if jaccards else 0
    else:
        avg_jaccard = 1.0
    
    results[strategy_name] = {
        'accuracy': float(overall_acc),
        'accuracy_std': float(np.std(fold_accuracies)),
        'macro_f1': float(macro_f1),
        'mean_n_features': float(np.mean(n_features_list)),
        'feature_selection_stability_jaccard': float(avg_jaccard),
        'confusion_matrix': cm.tolist(),
        'sensitivities': sensitivities,
    }
    
    print(f"    准确率: {overall_acc:.1%} (±{np.std(fold_accuracies):.1%})")
    print(f"    平均特征数: {np.mean(n_features_list):.0f}")
    print(f"    特征选择稳定性(Jaccard): {avg_jaccard:.3f}")
    print(f"    Macro F1: {macro_f1:.3f}")

# ========== 4. 汇总 ==========
print("\n" + "=" * 70)
print("结果汇总")
print("=" * 70)

comparison_table = []
for name, res in results.items():
    row = {
        'Strategy': name,
        'Accuracy': f"{res['accuracy']:.1%}",
        'Std': f"{res['accuracy_std']:.1%}",
        'N_features': f"{res['mean_n_features']:.0f}",
        'Stability': f"{res['feature_selection_stability_jaccard']:.3f}",
        'Macro_F1': f"{res['macro_f1']:.3f}",
        'AD_sens': f"{res['sensitivities'].get('AD', 0):.0%}",
        'CG_sens': f"{res['sensitivities'].get('CG', 0):.0%}",
    }
    comparison_table.append(row)

comparison_df = pd.DataFrame(comparison_table)
print("\n特征选择策略对比:")
print(comparison_df.to_string(index=False))

best_strategy = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n最优策略: {best_strategy[0]} (准确率: {best_strategy[1]['accuracy']:.1%})")

# ========== 保存 ==========
output = {
    'strategies': results,
    'comparison_table': comparison_table,
    'best_strategy': best_strategy[0],
    'best_accuracy': best_strategy[1]['accuracy'],
    'cv_method': '5 repeats x 10-fold stratified CV',
}

with open(f'{OUTPUT_DIR}\\analysis6_feature_selection.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

comparison_df.to_csv(f'{OUTPUT_DIR}\\analysis6_feature_selection.csv',
                      index=False, encoding='utf-8-sig')

print(f"\n结果已保存到: {OUTPUT_DIR}")
