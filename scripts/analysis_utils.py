# -*- coding: utf-8 -*-
"""
数据加载和预处理模块
加载表达矩阵、临床数据，执行药物过滤和年龄残差化
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ========== 路径配置 ==========
BASE_DIR = r'D:\冯熠的面上项目\SCI 论文攥写\第三篇文章\第三篇文章Final version\非靶向代谢组学分析'
EXP_PATH = f'{BASE_DIR}\\exp_download_express.csv'
CRF_PATH = f'{BASE_DIR}\\临床CRF表.xls'

# 注释列
ANNOTATION_COLS = ['ID', 'Metabolite', 'metab_id', 'Library ID', 'level', 'KEGG Compound ID',
                    'M/Z', 'Retention time', 'Mode', 'Adducts', 'Formula',
                    'Fragmentation Score', 'Theoretical Fragmentation Score', 'Mass Error',
                    'CAS ID', 'RSD']

# 疾病阶段编码（有序）
GROUP_ORDER = ['CG', 'SCD', 'MCI', 'AD']
GROUP_STAGE = {'CG': 0, 'SCD': 1, 'MCI': 2, 'AD': 3}


def load_expression_matrix():
    """加载表达矩阵，返回代谢物注释和表达值"""
    df = pd.read_csv(EXP_PATH)
    
    # 分离注释和表达值
    annot = df[ANNOTATION_COLS].copy()
    sample_cols = [c for c in df.columns if c not in ANNOTATION_COLS and not c.startswith('QC')]
    expr = df[sample_cols].copy()
    expr.index = annot['metab_id'].values
    
    # 转换为数值
    expr = expr.apply(pd.to_numeric, errors='coerce')
    
    # 填充缺失值为0（低于检测限）
    expr = expr.fillna(0)
    
    print(f"表达矩阵: {expr.shape[0]} 代谢物 × {expr.shape[1]} 样本")
    print(f"样本分组: {dict(pd.Series([c.split('_')[0] for c in expr.columns]).value_counts())}")
    
    return annot, expr


def load_clinical_data():
    """加载临床数据"""
    crf = pd.read_excel(CRF_PATH, engine='xlrd')
    
    # 用Sample Analysis Name作为索引（和表达矩阵列名对应）
    crf = crf.set_index('Sample Analysis Name')
    
    # 清理分组标签（去除前后空格）
    crf['Group'] = crf['Group'].astype(str).str.strip()
    
    # 编码分类变量
    crf['Gender_code'] = (crf['Gender'] == 'Male').astype(int)
    
    # 教育程度编码（有序）
    edu_map = {'Primary School': 1, 'Middle School': 2, 'High School': 3, 
               'College': 4, 'Bachelor': 4, 'Master': 5, 'PhD': 6}
    crf['Education_code'] = crf['Education'].map(edu_map).fillna(3)
    
    # 疾病阶段
    crf['Stage'] = crf['Group'].map(GROUP_STAGE)
    
    print(f"临床数据: {crf.shape[0]} 样本 × {crf.shape[1]} 变量")
    print(f"分组: {dict(crf['Group'].value_counts())}")
    print(f"年龄: {crf['Age'].mean():.1f} ± {crf['Age'].std():.1f}")
    
    return crf


def filter_by_level(annot, expr, level='B(i)'):
    """按鉴定水平过滤代谢物"""
    mask = annot['level'] == level
    filtered_annot = annot[mask].copy()
    filtered_expr = expr.loc[filtered_annot['metab_id'].values].copy()
    print(f"过滤为 {level}: {filtered_expr.shape[0]} 代谢物")
    return filtered_annot, filtered_expr


def log_transform(expr):
    """log2(x+1)转换"""
    return np.log2(expr + 1)


def age_residualization(expr_log, crf, covariates=['Age', 'Education_code', 'Gender_code']):
    """
    年龄残差化（线性回归方法）
    Residuals = log(abundance+1) - β_age*Age - β_edu*Edu - β_sex*Sex
    
    参数:
        expr_log: log转换后的表达矩阵 (代谢物 × 样本)
        crf: 临床数据
        covariates: 要校正的协变量列表
    
    返回:
        残差化后的表达矩阵
    """
    # 确保样本顺序一致
    common_samples = [s for s in expr_log.columns if s in crf.index]
    expr_aligned = expr_log[common_samples]
    crf_aligned = crf.loc[common_samples]
    
    # 协变量矩阵
    X = crf_aligned[covariates].values.astype(float)
    
    # 对每个代谢物拟合线性回归并取残差
    residuals = np.zeros_like(expr_aligned.values)
    for i in range(expr_aligned.shape[0]):
        y = expr_aligned.values[i, :]
        # 跳过全零或方差为0的代谢物
        if np.std(y) < 1e-10:
            residuals[i, :] = y
            continue
        model = LinearRegression()
        model.fit(X, y)
        residuals[i, :] = y - model.predict(X)
    
    result = pd.DataFrame(residuals, index=expr_aligned.index, columns=common_samples)
    print(f"年龄残差化完成: {result.shape[0]} 代谢物 × {result.shape[1]} 样本")
    print(f"  协变量: {covariates}")
    
    return result


def combat_adjust(expr_log, crf, batch_var='Age'):
    """
    ComBat批次校正（把年龄当批次变量）
    简化版ComBat：位置和尺度校正
    """
    common_samples = [s for s in expr_log.columns if s in crf.index]
    expr_aligned = expr_log[common_samples]
    crf_aligned = crf.loc[common_samples]
    
    # 将年龄分箱为批次（年轻/中年/老年）
    age_bins = pd.cut(crf_aligned['Age'], bins=3, labels=[0, 1, 2])
    batches = age_bins.values.astype(int)
    
    # 简化ComBat：每个批次内标准化，然后整体标准化
    result_arr = expr_aligned.values.astype(float).copy()
    for batch in np.unique(batches):
        mask = batches == batch
        if mask.sum() > 1:
            batch_mean = result_arr[:, mask].mean(axis=1, keepdims=True)
            batch_std = result_arr[:, mask].std(axis=1, keepdims=True) + 1e-10
            result_arr[:, mask] = (result_arr[:, mask] - batch_mean) / batch_std
    
    # 整体恢复到原始尺度
    overall_mean = expr_aligned.values.mean(axis=1, keepdims=True)
    overall_std = expr_aligned.values.std(axis=1, keepdims=True) + 1e-10
    result_arr = result_arr * overall_std + overall_mean
    
    result = pd.DataFrame(result_arr, index=expr_aligned.index, columns=common_samples)
    print(f"ComBat校正完成: {result.shape[0]} 代谢物 × {result.shape[1]} 样本")
    return result


def age_binning_standardize(expr_log, crf, n_bins=3):
    """
    年龄分箱后分层标准化
    在每个年龄箱内做z-score标准化
    """
    common_samples = [s for s in expr_log.columns if s in crf.index]
    expr_aligned = expr_log[common_samples]
    crf_aligned = crf.loc[common_samples]
    
    # 年龄分箱
    age_bins = pd.qcut(crf_aligned['Age'], q=n_bins, labels=range(n_bins), duplicates='drop')
    bins = age_bins.values
    
    # 每个箱内z-score标准化
    result = expr_aligned.copy().values.astype(float)
    for b in np.unique(bins):
        mask = bins == b
        if mask.sum() > 1:
            col_mean = result[:, mask].mean(axis=1, keepdims=True)
            col_std = result[:, mask].std(axis=1, keepdims=True) + 1e-10
            result[:, mask] = (result[:, mask] - col_mean) / col_std
    
    result = pd.DataFrame(result, index=expr_aligned.index, columns=common_samples)
    print(f"年龄分箱标准化完成: {result.shape[0]} 代谢物 × {result.shape[1]} 样本 (n_bins={n_bins})")
    return result


def get_sample_groups(expr, crf):
    """获取样本分组标签"""
    common_samples = [s for s in expr.columns if s in crf.index]
    groups = crf.loc[common_samples, 'Group'].values
    return common_samples, groups


if __name__ == '__main__':
    # 测试数据加载
    annot, expr = load_expression_matrix()
    crf = load_clinical_data()
    
    # 过滤B(i)级
    annot_bi, expr_bi = filter_by_level(annot, expr, 'B(i)')
    
    # log转换
    expr_log = log_transform(expr_bi)
    
    # 年龄残差化
    expr_resid = age_residualization(expr_log, crf)
    
    print("\n数据加载和预处理测试完成!")
