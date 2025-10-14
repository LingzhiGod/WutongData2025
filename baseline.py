#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import gc
import math
import glob
import json
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------------------------
# 一、工具函数：文件发现与加载
# ------------------------------

def find_file(candidates: List[str]) -> str:
    for pat in candidates:
        files = glob.glob(pat)
        if files:
            return files[0]
    return ""


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """尝试多种常见命名以提升兼容性。"""
    train_fp = find_file(["train.csv", "./data/train.csv", "./Train.csv", "./train_*.csv"]) \
        or find_file(["*train*.csv"])  # 兜底

    test_fp = find_file(["test.csv", "./data/test.csv", "./TestA.csv", "./testA.csv", "./test_*.csv"]) \
        or find_file(["*test*.csv", "*A.csv"])  # 兜底

    if not train_fp or not test_fp:
        print("[ERROR] 未找到 train/test CSV，请将脚本与数据放在同一目录或调整上面的匹配模式。")
        sys.exit(1)

    print(f"[INFO] 读取训练集: {train_fp}")
    print(f"[INFO] 读取测试集: {test_fp}")

    train = pd.read_csv(train_fp)
    test = pd.read_csv(test_fp)

    # 统一列名中的空格等问题
    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]

    return train, test


# ------------------------------
# 二、特征工程
# ------------------------------

CAT_COLS_RAW = [
    "gender",              # 1男 2女（后续转0/1）
    "uses_education_app",  # 0/1
    "uses_entertainment_app",
    "uses_shopping_app",
    "residence_base_station_id",  # 高基数ID，树模型可直接做类别特征
    "residence_cell_id",
    "tariff_id",
]

NUM_COLS_RAW = [
    "age",
    "registration_channel_id",
    "over_limit_data(MB)",
    "tariff_price(RMB)",
    "total_data(MB)",
    "total_voice(minutes)",
    "call_duration(minutes)",
    "monthly_call_count",
    "monthly_weekend_call_count",
    "avg_call_duration(minutes)",
    "avg_weekday_call_duration(minutes)",
    "avg_weekend_call_duration(minutes)",
    "residence_duration_9to11",
    "residence_duration_11to14",
    "residence_duration_14to17",
    "residence_duration_17to21",
    "residence_duration_21to23",
    "residence_duration_24to6",
    "total_residence_duration",
]

DATE_COL = "registration_date"
ID_COL = "user_id"
TARGET = "is_positive"


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def safe_div(a, b):
    return a / np.where(b == 0, 1, b)


def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()

    # 1) 类型规范化
    # 日期：转为“入网距今天数”与年/月等
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        ref_date = pd.to_datetime("2020-01-01")  # 固定参考点，避免泄漏
        df["reg_days_since_ref"] = (df[DATE_COL] - ref_date).dt.days.astype("float32")
        df["reg_year"] = df[DATE_COL].dt.year
        df["reg_month"] = df[DATE_COL].dt.month
    else:
        df["reg_days_since_ref"] = np.nan
        df["reg_year"] = np.nan
        df["reg_month"] = np.nan

    # 数值列转数值
    coerce_numeric(df, NUM_COLS_RAW)

    # 类别列：先保证字符串化（便于统一编码/类别设置）
    for c in CAT_COLS_RAW:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # 2) 缺失填充（简单策略，后续可改进）
    for c in NUM_COLS_RAW + ["reg_days_since_ref"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    for c in CAT_COLS_RAW + ["reg_year", "reg_month"]:
        if c in df.columns:
            df[c] = df[c].fillna("-1")

    # 3) 原子派生特征
    # 性别/APP使用 转 0/1
    if "gender" in df.columns:
        df["gender_bin"] = df["gender"].map({"1": 1, "2": 0}).fillna(0).astype(int)
    for c in ["uses_education_app", "uses_entertainment_app", "uses_shopping_app"]:
        if c in df.columns:
            df[c + "_bin"] = df[c].map({"1": 1, "0": 0}).fillna(0).astype(int)

    # 时间段占比（夜间/白天/晚间）
    r9_11 = df.get("residence_duration_9to11", 0)
    r11_14 = df.get("residence_duration_11to14", 0)
    r14_17 = df.get("residence_duration_14to17", 0)
    r17_21 = df.get("residence_duration_17to21", 0)
    r21_23 = df.get("residence_duration_21to23", 0)
    r24_6 = df.get("residence_duration_24to6", 0)
    total_r = (df.get("total_residence_duration", 0)).replace(0, 1)

    df["ratio_night"] = safe_div((r21_23 + r24_6), total_r)
    df["ratio_evening"] = safe_div(r17_21, total_r)
    df["ratio_day"] = safe_div((r9_11 + r11_14 + r14_17), total_r)

    # 通话与语音强度
    call_cnt = df.get("monthly_call_count", 0).replace(0, 1)
    df["avg_call_dur_calc"] = safe_div(df.get("call_duration(minutes)", 0), call_cnt)
    df["weekend_call_ratio"] = safe_div(df.get("monthly_weekend_call_count", 0), df.get("monthly_call_count", 0).replace(0, 1))
    df["voice_per_call"] = safe_div(df.get("total_voice(minutes)", 0), call_cnt)

    # 资费/流量性价比
    df["price_per_mb"] = safe_div(df.get("tariff_price(RMB)", 0), df.get("total_data(MB)", 0).replace(0, 1))

    # 日间/夜间交互与年龄交互
    df["young_night"] = df.get("age", 0) * df["ratio_night"]
    df["edu_pref"] = df.get("uses_education_app_bin", 0) * (1 - df.get("uses_shopping_app_bin", 0))

    # 总时段一致性检查（可作为异常信号）
    sum_slots = r9_11 + r11_14 + r14_17 + r17_21 + r21_23 + r24_6
    df["slot_total_match"] = np.isclose(sum_slots.values, df.get("total_residence_duration", 0).values, rtol=0.05, atol=5).astype(int)

    # 4) 高基数ID直接保留为类别（树模型可原生支持类别/或做LabelEncode）
    # 为了通用性，这里统一做 LabelEncode，线性模型也能用；
    # 若使用LightGBM且想用原生类别，后面会再把这些列转为"category"。
    lbl_map: Dict[str, LabelEncoder] = {}
    for c in ["residence_base_station_id", "residence_cell_id", "tariff_id", "gender", "uses_education_app", "uses_entertainment_app", "uses_shopping_app", "reg_year", "reg_month"]:
        if c in df.columns:
            le = LabelEncoder()
            df[c + "_le"] = le.fit_transform(df[c].astype(str))
            lbl_map[c] = le

    # 5) 选择最终用于建模的列
    #   - 数值列 + 派生列 + LabelEncode 后的类别列
    used_cols = []
    used_cols += [c for c in NUM_COLS_RAW if c in df.columns]
    used_cols += [
        "reg_days_since_ref", "reg_year_le", "reg_month_le",
        "gender_bin", "uses_education_app_bin", "uses_entertainment_app_bin", "uses_shopping_app_bin",
        "ratio_night", "ratio_evening", "ratio_day",
        "avg_call_dur_calc", "weekend_call_ratio", "voice_per_call",
        "price_per_mb", "young_night", "edu_pref", "slot_total_match",
        "residence_base_station_id_le", "residence_cell_id_le", "tariff_id_le",
    ]
    used_cols = [c for c in used_cols if c in df.columns]

    df["__used_cols__"] = ",".join(used_cols)
    return df


# ------------------------------
# 三、模型训练与预测（优先 LightGBM → 其次 XGBoost → 回退 Logistic）
# ------------------------------

def fit_predict_with_lgb(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    import lightgbm as lgb

    oof_pred = np.zeros(len(train_df))
    tst_pred = np.zeros(len(test_df))
    fi_list = []
    thresholds = []

    X = train_df[features]
    y = train_df[TARGET].astype(int).values
    X_test = test_df[features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    params = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=20,
        seed=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        lgb_trn = lgb.Dataset(X_trn, label=y_trn)
        lgb_val = lgb.Dataset(X_val, label=y_val)

        clf = lgb.train(
            params,
            lgb_trn,
            num_boost_round=2000,
            valid_sets=[lgb_trn, lgb_val],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(100),
            ],
        )

        # 验证集预测 + 阈值搜索（按F1最大化）
        val_prob = clf.predict(X_val, num_iteration=clf.best_iteration)
        oof_pred[val_idx] = val_prob

        # 阈值扫描
        thr_candidates = np.linspace(0.2, 0.8, 61)  # 0.2~0.8 步长0.01
        f1s = []
        for thr in thr_candidates:
            f1s.append(f1_score(y_val, (val_prob >= thr).astype(int)))
        best_thr = float(thr_candidates[int(np.argmax(f1s))])
        thresholds.append(best_thr)
        print(f"[Fold {fold}] best F1={max(f1s):.5f} @ thr={best_thr:.3f}")

        # 测试集预测累计
        tst_prob = clf.predict(X_test, num_iteration=clf.best_iteration)
        tst_pred += tst_prob / skf.n_splits

        # 特征重要性
        fi = pd.DataFrame({
            "feature": features,
            "importance": clf.feature_importance(importance_type="gain"),
            "fold": fold,
        })
        fi_list.append(fi)

        del clf, lgb_trn, lgb_val
        gc.collect()

    oof_thr = float(np.mean(thresholds))
    print(f"[OOF] 使用平均阈值 thr={oof_thr:.3f}")

    fi_df = pd.concat(fi_list, axis=0, ignore_index=True)
    return oof_pred, tst_pred, fi_df, oof_thr


def fit_predict_with_xgb(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]):
    import xgboost as xgb

    oof_pred = np.zeros(len(train_df))
    tst_pred = np.zeros(len(test_df))
    fi_list = []
    thresholds = []

    X = train_df[features]
    y = train_df[TARGET].astype(int).values
    X_test = test_df[features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    params = dict(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=RANDOM_STATE,
        nthread=-1,
    )

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        dtrn = xgb.DMatrix(X_trn, label=y_trn)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        clf = xgb.train(
            params,
            dtrn,
            num_boost_round=5000,
            evals=[(dtrn, "train"), (dval, "valid")],
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(200),
            ],

        )

        val_prob = clf.predict(dval, ntree_limit=clf.best_ntree_limit)
        oof_pred[val_idx] = val_prob

        thr_candidates = np.linspace(0.2, 0.8, 61)
        f1s = [f1_score(y_val, (val_prob >= thr).astype(int)) for thr in thr_candidates]
        best_thr = float(thr_candidates[int(np.argmax(f1s))])
        thresholds.append(best_thr)
        print(f"[Fold {fold}] best F1={max(f1s):.5f} @ thr={best_thr:.3f}")

        tst_prob = clf.predict(dtest, ntree_limit=clf.best_ntree_limit)
        tst_pred += tst_prob / skf.n_splits

        # XGBoost 不易直接拿到特征重要性（gain）按Booster处理：
        score = clf.get_score(importance_type='gain')
        fi = pd.DataFrame({
            "feature": list(score.keys()),
            "importance": list(score.values()),
            "fold": fold,
        })
        fi_list.append(fi)

        del clf, dtrn, dval, dtest
        gc.collect()

    oof_thr = float(np.mean(thresholds))
    fi_df = pd.concat(fi_list, axis=0, ignore_index=True)
    return oof_pred, tst_pred, fi_df, oof_thr


def fit_predict_with_logreg(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # 线性模型需要缩放
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1))
    ])

    oof_pred = np.zeros(len(train_df))
    tst_pred = np.zeros(len(test_df))
    thresholds = []

    X = train_df[features]
    y = train_df[TARGET].astype(int).values
    X_test = test_df[features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        pipe.fit(X_trn, y_trn)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        oof_pred[val_idx] = val_prob

        thr_candidates = np.linspace(0.2, 0.8, 61)
        f1s = [f1_score(y_val, (val_prob >= thr).astype(int)) for thr in thr_candidates]
        best_thr = float(thr_candidates[int(np.argmax(f1s))])
        thresholds.append(best_thr)
        print(f"[Fold {fold}] best F1={max(f1s):.5f} @ thr={best_thr:.3f}")

        tst_prob = pipe.predict_proba(X_test)[:, 1]
        tst_pred += tst_prob / skf.n_splits

    return oof_pred, tst_pred, pd.DataFrame(), float(np.mean(thresholds))


# ------------------------------
# 四、主流程
# ------------------------------

def main():
    train, test = load_data()

    assert ID_COL in train.columns, f"训练集缺少 {ID_COL}"
    assert ID_COL in test.columns, f"测试集缺少 {ID_COL}"
    assert TARGET in train.columns, f"训练集缺少标签列 {TARGET}"

    # 构建特征
    train_fe = build_features(train, is_train=True)
    test_fe = build_features(test, is_train=False)

    used_cols = train_fe["__used_cols__"].iloc[0].split(",")
    used_cols = [c for c in used_cols if c in train_fe.columns]

    print(f"[INFO] 使用 {len(used_cols)} 个特征：\n{used_cols}\n")

    # 选择模型：优先 LightGBM → 其次 XGBoost → 回退 Logistic
    oof, pred, fi_df, thr = None, None, None, 0.5

    try:
        import lightgbm  # type: ignore
        print("[INFO] 使用 LightGBM 训练...")
        oof, pred, fi_df, thr = fit_predict_with_lgb(train_fe, test_fe, used_cols)
    except Exception as e_lgb:
        print(f"[WARN] LightGBM 不可用，回退 XGBoost: {e_lgb}")
        try:
            import xgboost  # type: ignore
            print("[INFO] 使用 XGBoost 训练...")
            oof, pred, fi_df, thr = fit_predict_with_xgb(train_fe, test_fe, used_cols)
        except Exception as e_xgb:
            print(f"[WARN] XGBoost 不可用，回退 LogisticRegression: {e_xgb}")
            print("[INFO] 使用 LogisticRegression 训练...")
            oof, pred, fi_df, thr = fit_predict_with_logreg(train_fe, test_fe, used_cols)

    # 评估OOF
    y_true = train[TARGET].astype(int).values
    oof_label = (oof >= thr).astype(int)
    acc = accuracy_score(y_true, oof_label)
    f1 = f1_score(y_true, oof_label)
    pre = precision_score(y_true, oof_label)
    rec = recall_score(y_true, oof_label)
    print(f"[OOF] Acc={acc:.5f}  F1={f1:.5f}  P={pre:.5f}  R={rec:.5f}  Thr={thr:.3f}")

    # 生成提交
    sub = pd.DataFrame({
        ID_COL: test[ID_COL].values,
        TARGET: (pred >= thr).astype(int)
    })
    sub.to_csv("submission.csv", index=False)
    print("[OK] 已生成 submission.csv")

    # 保存特征重要性
    if fi_df is not None and not fi_df.empty:
        fi_agg = fi_df.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
        fi_agg.to_csv("feature_importance.csv", index=False)
        print("[OK] 已生成 feature_importance.csv（gain 平均值）")


if __name__ == "__main__":
    main()
