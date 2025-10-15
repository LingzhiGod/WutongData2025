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
        "metric": "binary_logloss",
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
        "n_jobs": -1,
        "seed": ver2.RANDOM_STATE,
        "verbose": -1,
    }

    # 2️⃣ 加载数据 + 特征工程
    train_df, test_df = ver2.load_data()
    train_df = ver2.build_features(train_df, is_train=True)
    test_df = ver2.build_features(test_df, is_train=False)

    used_cols = train_df["__used_cols__"].iloc[0].split(",")

    # 3️⃣ 准备数据
    X = train_df[used_cols]
    y = train_df[ver2.TARGET].astype(int).values
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=ver2.RANDOM_STATE)

    oof_pred = np.zeros(len(y))
    thresholds = []

    # 4️⃣ 交叉验证
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]

        dtrain = lgb.Dataset(X_trn, label=y_trn)
        dvalid = lgb.Dataset(X_val, label=y_val)

        clf = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dvalid],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(200),
            ],
        )

        val_prob = clf.predict(X_val, num_iteration=clf.best_iteration)
        oof_pred[val_idx] = val_prob

        # 阈值扫描
        thr_candidates = np.linspace(0.2, 0.8, 61)
        f1s = [f1_score(y_val, (val_prob >= thr).astype(int)) for thr in thr_candidates]
        thresholds.append(thr_candidates[np.argmax(f1s)])

        del clf, dtrain, dvalid
        gc.collect()

    avg_thr = np.mean(thresholds)
    f1 = f1_score(y, (oof_pred >= avg_thr).astype(int))
    print(f"[Trial] mean F1={f1:.5f}  avg_thr={avg_thr:.3f}")
    return f1


# ===================================================
# 主函数入口
# ===================================================
def main():
    print("🚀 Starting LightGBM parameter tuning using Optuna ...")
    study = optuna.create_study(direction="maximize", study_name="lgbm_tune")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("\n=================== 最优结果 ===================")
    print(f"✅ 最优F1: {study.best_value:.5f}")
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
