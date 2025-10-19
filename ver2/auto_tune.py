#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna 自动调参脚本（基于 ver2.py）
---------------------------------------------------
功能：
1. 自动导入 ver2.py 中的数据加载、特征工程与模型结构；
2. 使用 5 折交叉验证；
3. 以 F1-score 为目标自动搜索 LightGBM 超参数；
4. 输出最优结果与参数文件。
---------------------------------------------------
使用方法：
    python tune_lgbm_optuna.py
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb

# 导入核心脚本 ver2.py 中的定义
import ver2
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


# ===================================================
# Optuna 调参目标函数
# ===================================================
def objective(trial):
    # 1️⃣ 定义搜索空间
    params = {
        "objective": "binary",
        "metric": ["binary_logloss","auc","average_precision"],
        #"is_unbalance": True,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, step=16),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0, step=0.05),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0, step=0.05),
        "bagging_freq": 1,
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 1),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 3),
        "max_bin": trial.suggest_int("max_bin", 128, 512),
        "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        "extra_trees_frequency": trial.suggest_int("extra_trees_frequency", 1, 10),
        "boost_from_average": trial.suggest_categorical("boost_from_average", [True, False]),
        "n_jobs": -1,
        "seed": ver2.RANDOM_STATE,
        "verbose": -1,
    }

    # 2️⃣ 加载数据 + 特征工程
    train_df, test_df = ver2.load_data()
    train_df = ver2.build_features(train_df, is_train=True)
    test_df = ver2.build_features(test_df, is_train=False)

    used_cols = train_df["__used_cols__"].iloc[0].split(",")
    cat_le_cols = train_df["__cat_le_cols__"].iloc[0].split(",")
    num_cols = train_df["__num_cols__"].iloc[0].split(",")

    # 3️⃣ 准备数据
    X = train_df[used_cols]
    y = train_df[ver2.TARGET].astype(int).values
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=ver2.RANDOM_STATE)

    oof_pred = np.zeros(len(y))

    # 4️⃣ 交叉验证
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]

        dtrain = lgb.Dataset(X_trn, label=y_trn, categorical_feature=cat_le_cols, free_raw_data=False)
        dvalid = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_le_cols, reference=dtrain, free_raw_data=False)

        clf = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dvalid],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(200),
            ],
        )

        val_prob = clf.predict(X_val, num_iteration=clf.best_iteration)
        oof_pred[val_idx] = val_prob
        del clf, dtrain, dvalid
        gc.collect()

    oof_thr, best_score, values = ver2.pick_best_threshold_by_score(y, oof_pred, step=0.01)
    f1 = values[1]
    print(f"[Trial] best F1={f1:.5f}  best_thr={oof_thr:.3f}")
    return best_score


# ===================================================
# 主函数入口
# ===================================================\
from optuna.samplers import TPESampler
def main():
    print("🚀 Starting LightGBM parameter tuning using Optuna ...")
    study = optuna.create_study(direction="maximize", study_name="lgbm_tune",sampler= TPESampler(multivariate=True, group=True, seed=ver2.RANDOM_STATE))
    study.optimize(objective, n_trials=60, show_progress_bar=True)

    print("\n=================== 最优结果 ===================")
    print(f"✅ 最优Score: {study.best_value:.5f}")
    print("🏆 最优参数组合:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 保存结果
    df_best = pd.DataFrame([study.best_params])
    df_best["best_f1"] = study.best_value
    df_best.to_csv("optuna_best_params.csv", index=False)
    print("💾 最优参数已保存至 optuna_best_params.csv")


if __name__ == "__main__":
    main()
