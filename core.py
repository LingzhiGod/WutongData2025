import os
import sys
import gc
import glob
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# --------------------------------------
# Constant Pool
# --------------------------------------
TRAIN_FP = "./train.csv"
TEST_FP = "./test.csv"

ID_COL = "user_id"
DATE_COL = "registration_date"
TARGET = "is_positive"
REG_YEAR = "reg_year"
REG_MONTH = "reg_month"
REG_DAY_SINCE_REF = "reg_days_since_ref"

CAT_COLS_RAW = [
    "registration_channel_id",
    "gender",  # 1男 2女（后续转0/1）
    "uses_education_app",  # 0/1
    "uses_entertainment_app",
    "uses_shopping_app",
    "residence_base_station_id",  # 高基数ID，树模型可直接做类别特征
    "residence_cell_id",
    "tariff_id"
]

NUM_COLS_RAW = [
    "age",
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

REF_DATE = pd.to_datetime("2020-12-31")
RANDOM_STATE = 20070702
np.random.seed(RANDOM_STATE)
# --------------------------------------

lgbm_default_params = dict(
    objective="binary",
    metric=["binary_logloss", "auc","average_precision"],
    first_metric_only=True,
    boosting_type="gbdt",
    is_unbalance=True,
    learning_rate=0.025,

    num_leaves=31,
    max_depth=5,
    min_data_in_leaf=225,
    min_split_gain=0.45,


    # feature_fraction=0.85,
    # bagging_fraction=0.85,
    bagging_freq=1,

    # # ===== 正则化 =====
    # lambda_l1=1.5,
    # lambda_l2=0.85,

    seed=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)



def load_best_params(default_params: dict, filename: str = "optuna_best_params.csv") -> dict:
    if not os.path.exists(filename):
        print(f"[Optuna]{filename} not found,using default params.")
        return default_params

    try:
        best_df = pd.read_csv(filename)
        best_params = best_df.iloc[0].to_dict()
        best_f1 = best_params.pop("best_f1", None)

        int_keys = {"num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq"}

        for k, v in best_params.items():
            if isinstance(v, str):
                try:
                    if v.lower() in ["true", "false"]:
                        best_params[k] = v.lower() == "true"
                    elif "." in v:
                        best_params[k] = float(v)
                    else:
                        best_params[k] = int(v)
                except Exception:
                    pass
            elif isinstance(v, float) and k in int_keys:
                best_params[k] = int(v)

        updated = default_params.copy()
        updated.update(best_params)

        print(f"[Optuna]Loaded ({filename})")
        if best_f1 is not None:
            print(f"[Optuna]Best F1 = {best_f1:.5f}")
        print(f"[Optuna]Loaded params：{list(best_params.keys())}")

        return updated
    except Exception as e:
        print(f"[Optuna]Failed when loading best param：{e}")
        return default_params

def fit_predict_with_lgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str], cat_features: List[str],
                          num_features: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    import lightgbm as lgb

    oof_pred = np.zeros(len(train_df))
    tst_pred = np.zeros(len(test_df))

    fi_list = []

    X = train_df[features]
    y = train_df[TARGET].astype(int).values
    X_test = test_df[features]

    lgbm_params = load_best_params(lgbm_default_params)

    lgbm_params.pop("is_unbalance", None)
    lgbm_params["scale_pos_weight"] = compute_scale(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_ids = np.zeros(len(train_df), dtype=int)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        fold_ids[val_idx] = fold

        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        lgb_trn = lgb.Dataset(X_trn, label=y_trn, categorical_feature=cat_features, free_raw_data=False)
        lgb_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=lgb_trn,
                              free_raw_data=False)

        clf = lgb.train(
            lgbm_params,
            lgb_trn,
            num_boost_round=2000,
            valid_sets=[lgb_trn, lgb_val],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(20),
            ],
        )

        # 验证集预测 + 阈值搜索（按F1最大化）
        val_prob = clf.predict(X_val, num_iteration=clf.best_iteration)
        oof_pred[val_idx] = val_prob

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

    fi_df = pd.concat(fi_list, axis=0, ignore_index=True)
    return oof_pred, tst_pred, fi_df, fold_ids


# --------------------------------------
# Util functions
# --------------------------------------
def find_file(candidates: List[str]) -> str:
    for pat in candidates:
        files = glob.glob(pat)
        if files:
            return files[0]
    return ""

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_fp = find_file(TRAIN_FP)
    test_fp = find_file(TEST_FP)

    print("[Info]Loading data...")

    if not train_fp or not test_fp:
        print("[ERROR] train/test csv not found。")
        sys.exit(1)

    train = pd.read_csv(TRAIN_FP)
    test = pd.read_csv(TEST_FP)

    return train, test

def compute_scale(y: ndarray, eps: float = 1e-6) -> float:
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0:
        print("[WARNING]No positive samples found when computing scale. Return 1.0")
        return 1.0
    spw = neg / (pos + eps)
    print(f"[INFO]Scale = {spw:.4f}  (neg={neg}, pos={pos})")
    return float(spw)

def safe_div(a, b):
    return a / np.where(b == 0, 1, b)

def wow(a, b, total, which):
    if total == 0:
        return 0.5
    return np.where(which, a, b) / total

def platform_score(y_true, y_pred_bin):
    acc = accuracy_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    return 0.7 * acc + 0.3 * f1

def choose_threshold_generalization(
    y_true: np.ndarray,
    prob: np.ndarray,
    fold_ids: np.ndarray,
    thr_min: float = 0.05,
    thr_max: float = 0.95,
    thr_step: float = 0.001,
    delta_f1: float = 0.002,
    alpha_pos: float = 1.0,   # 阳性率偏离惩罚
    beta_var: float = 0.5,    # 折间 F1 方差惩罚
    pos_rate_target: float | None = None,
):
    """
    泛化优先的阈值选择：
    1) 先找全局 F1 最大的 F1_max
    2) 只在 F1 >= F1_max - delta_f1 的“高 F1 区间”里选阈值
    3) 在该区间中综合考虑：
       - 阳性率离 pos_rate_target 的距离（越近越好）
       - 各折 F1 的方差（越小越好）
       - 全局 F1（越高越好）
    """
    if pos_rate_target is None:
        pos_rate_target = float(np.mean(y_true))
        print(f"[GenThr] pos_rate_target not given, use train pos_rate={pos_rate_target:.5f}")

    ths = np.arange(thr_min, thr_max + 1e-9, thr_step)

    # 先算每个阈值的全局 F1
    f1_list = []
    for t in ths:
        pred = (prob >= t).astype(int)
        f1_glb = f1_score(y_true, pred, zero_division=0)
        f1_list.append(f1_glb)

    f1_max = max(f1_list)

    # 高 F1 阈值集合
    high_f1_ths = [t for t, f1_g in zip(ths, f1_list) if f1_g >= f1_max - delta_f1]
    if not high_f1_ths:
        high_f1_ths = list(ths)  # 兜底

    best = None

    for t in high_f1_ths:
        pred = (prob >= t).astype(int)
        f1_glb = f1_score(y_true, pred, zero_division=0)
        pos_rate = pred.mean()

        # 各折 F1
        f1_folds = []
        for fold in np.unique(fold_ids):
            idx = (fold_ids == fold)
            if idx.sum() == 0:
                continue
            f1_k = f1_score(y_true[idx], pred[idx], zero_division=0)
            f1_folds.append(f1_k)
        var_f1 = np.var(f1_folds) if len(f1_folds) > 1 else 0.0

        # 惩罚：阳性率偏离 + 折间方差
        penalty = alpha_pos * abs(pos_rate - pos_rate_target) + beta_var * var_f1
        eff = f1_glb - penalty

        if (best is None) or (eff > best["eff"]):
            best = dict(
                thr=t,
                eff=eff,
                f1_glb=f1_glb,
                pos_rate=pos_rate,
                var_f1=var_f1,
                penalty=penalty,
            )

    print(
        "[GenThr] thr={thr:.5f} eff={eff:.5f} F1_glb={f1_glb:.5f} "
        "pos_rate={pos_rate:.5f} var_F1={var_f1:.5f} penalty={penalty:.5f} "
        "(F1_max={f1_max:.5f}, delta_f1={delta_f1:.4f})"
        .format(f1_max=f1_max, delta_f1=delta_f1, **best)
    )
    return best["thr"], best


def pick_best_threshold_by_score(y_true, prob, step=0.005):
    ths = np.arange(0.05, 0.95 + 1e-9, step)
    best_thr, best_score = 0, -1
    best_acc, best_f1, best_p, best_r = 0, 0, 0, 0

    for t in ths:
        pred = (prob >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, zero_division=0)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred)
        score = 0.7 * acc + 0.3 * f1

        if score > best_score:
            best_score = score
            best_thr = t
            best_acc, best_f1, best_p, best_r = acc, f1, p, r

    print(f"[THR Search] Best Score={best_score:.5f}  Acc={best_acc:.5f}  "
          f"F1={best_f1:.5f}  P={best_p:.5f}  R={best_r:.5f}  Thr={best_thr:.3f}")
    return best_thr, best_score, (best_acc, best_f1, best_p, best_r)


# --------------------------------------
# Feature Buildup
# --------------------------------------

encoders = {}

def build_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()
    local_cat_cols = CAT_COLS_RAW.copy()
    local_num_cols = NUM_COLS_RAW.copy()

    # Process Date
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df[REG_YEAR] = df[DATE_COL].dt.year
        df[REG_MONTH] = df[DATE_COL].dt.month
        df[REG_DAY_SINCE_REF] = (df[DATE_COL] - REF_DATE).dt.days.astype("float32")  # 手动转换规定转换行为避免不可预料的错误
        local_cat_cols.append(REG_YEAR)
        local_cat_cols.append(REG_MONTH)
        local_num_cols.append(REG_DAY_SINCE_REF)

    else:
        df[REG_YEAR] = np.nan
        df[REG_MONTH] = np.nan
        df[REG_DAY_SINCE_REF] = np.nan

    # Type Convert
    for name in NUM_COLS_RAW:
        if name in df.columns:
            df[name] = pd.to_numeric(df[name], errors="coerce")
    for name in CAT_COLS_RAW:
        if name in df.columns:
            df[name] = df[name].astype(str)

    avg_call = df.get("call_duration(minutes)", 0)
    total_call_count = df.get("monthly_call_count", 0)
    total_call = avg_call * total_call_count
    df["total_call_duration"] = total_call

    local_num_cols.append("total_call_duration")

    weekend_call_count = df.get("monthly_weekend_call_count", 0)
    weekday_call_count = total_call_count - weekend_call_count
    df["monthly_weekday_call_count"] = weekday_call_count
    local_num_cols.append("monthly_weekday_call_count")

    ratio_weekday_call = np.where(total_call_count == 0, 0.5, weekday_call_count / total_call_count)
    ratio_weekend_call = np.where(total_call_count == 0, 0.5, weekend_call_count / total_call_count)
    df["ratio_weekday_call"] = ratio_weekday_call
    df["ratio_weekend_call"] = ratio_weekend_call
    local_num_cols.append("ratio_weekday_call")
    local_num_cols.append("ratio_weekend_call")

    avg_weekday_call_dura = df.get("avg_weekday_call_duration(minutes)", 0)
    avg_weekend_call_dura = df.get("avg_weekend_call_duration(minutes)", 0)
    ratio_weekday_call_dura = np.where(total_call == 0, 0.5, avg_weekday_call_dura * weekday_call_count / total_call)
    ratio_weekend_call_dura = np.where(total_call == 0, 0.5, avg_weekend_call_dura * weekend_call_count / total_call)
    df["ratio_weekday_call_dura"] = ratio_weekday_call_dura
    df["ratio_weekend_call_dura"] = ratio_weekend_call_dura
    local_num_cols.append("ratio_weekday_call_dura")
    local_num_cols.append("ratio_weekend_call_dura")

    # Price per MB (robust handling when total_data == 0)
    price = pd.to_numeric(df.get("tariff_price(RMB)", 0), errors="coerce")
    data_mb = pd.to_numeric(df.get("total_data(MB)", 0), errors="coerce")
    # 当总流量为 0 时，该值不可定义：设为 NaN 以让树模型按缺失处理，并配合 no_data_allowance_flag 提示语义
    df["price_per_mb"] = np.where(data_mb > 0, price / data_mb, np.nan)
    local_num_cols.append("price_per_mb")

    df["no_data_allowance_flag"] = (data_mb <= 0).astype(int)
    local_cat_cols.append("no_data_allowance_flag")

    R9_11 = df.get("residence_duration_9to11", 0)
    R11_14 = df.get("residence_duration_11to14", 0)
    R14_17 = df.get("residence_duration_14to17", 0)
    R17_21 = df.get("residence_duration_17to21", 0)
    R21_23 = df.get("residence_duration_21to23", 0)
    R24_6 = df.get("residence_duration_24to6", 0)
    total_res = df.get("total_residence_duration", 0)

    df["ratio_night"] = np.where(total_res == 0, 1 / 3, (R24_6 + R21_23) / total_res)
    df["ratio_evening"] = np.where(total_res == 0, 1 / 3, R17_21 / total_res)
    df["ratio_day"] = np.where(total_res == 0, 1 / 3, (R9_11 + R11_14 + R14_17) / total_res)
    local_num_cols.append("ratio_night")
    local_num_cols.append("ratio_evening")
    local_num_cols.append("ratio_day")

    durations = df[
        [f"residence_duration_{p}" for p in ["9to11", "11to14", "14to17", "17to21", "21to23", "24to6"]]].fillna(
        0).values
    p = durations / (durations.sum(axis=1, keepdims=True) + 1e-6)
    df["res_duration_entropy"] = -(p * np.log(p + 1e-9)).sum(axis=1)
    local_num_cols.append("res_duration_entropy")
    #
    # uniform = np.full_like(p, 1/len(durations))
    # df["residence_kl_to_uniform"] = np.sum(p * np.log((p+1e-9)/(uniform+1e-9)), axis=1)
    # df["residence_gini"] = 1 - np.sum(p**2, axis=1)
    # local_num_cols.append("residence_kl_to_uniform")
    # local_num_cols.append("residence_gini")
    #
    # df["residence_symmetry"] = 1 - np.abs(df["ratio_day"] - df["ratio_night"])
    # local_num_cols.append("residence_symmetry")

    df["day_night_square_diff"] = (df["ratio_day"] - df["ratio_night"]) ** 2
    local_num_cols.append("day_night_square_diff")

    df["day_night_res_diff"] = df["ratio_day"] - df["ratio_night"]
    df["evening_day_diff"] = df["ratio_evening"] - df["ratio_day"]
    local_num_cols.append("day_night_res_diff")
    local_num_cols.append("evening_day_diff")

    df["is_school_open_season"] = df["reg_month"].isin([3, 9]).astype(int)
    local_cat_cols.append("is_school_open_season")
    # Label Encoding
    if is_train:
        for cat in local_cat_cols:
            if cat in df.columns:
                encoder = LabelEncoder()
                df[cat + "_le"] = encoder.fit_transform(df[cat].astype(str))
                encoders[cat] = encoder
    else:
        for cat in local_cat_cols:
            if cat in df.columns:
                encoder = encoders.get(cat)
                if encoder is not None:
                    unseen = set(df[cat].astype(str)) - set(encoder.classes_)
                    if unseen:
                        encoder.classes_ = np.append(encoder.classes_,
                                                     list(unseen))  # New value as UNK Label for code robust
                    df[cat + "_le"] = encoder.transform(df[cat].astype(str))
    using_cols = []
    using_cols += [n for n in local_num_cols if n in df.columns]
    using_cols += [c + "_le" for c in local_cat_cols if c in df.columns]
    print("[Info]Features built, using " + str(len(using_cols)) + " features:", using_cols)
    df["__used_cols__"] = ",".join(using_cols)
    df["__cat_le_cols__"] = ",".join([c + "_le" for c in local_cat_cols if c in df.columns])
    df["__num_cols__"] = ",".join([n for n in local_num_cols if n in df.columns])
    return df


def main():
    train, test = load_data()

    train_fe = build_features(train, is_train=True)
    test_fe = build_features(test, is_train=False)

    using_cols = train_fe["__used_cols__"].iloc[0].split(",")
    cat_le_cols = train_fe["__cat_le_cols__"].iloc[0].split(",")
    num_cols = train_fe["__num_cols__"].iloc[0].split(",")

    try:
        import lightgbm
        print("[INFO]Training LGBM...")
    except Exception as e_lgb:
        print("[ERROR] Failed when training by using lightgbm.")
        print("[ERROR] Error: " + str(e_lgb))

    oof, pred, fi_df, fold_ids = fit_predict_with_lgbm(
        train_fe, test_fe, using_cols, cat_le_cols, num_cols
    )

    y_true = train[TARGET].astype(int).values

    thr, info = choose_threshold_generalization(
        y_true=y_true,
        prob=oof,
        fold_ids=fold_ids,
        thr_min=0.05,
        thr_max=0.95,
        thr_step=0.001,
        delta_f1=0.002,
        alpha_pos=1.0,
        beta_var=0.5,
        pos_rate_target=6500 / 43500,  # 自动用 train 里的正例率≈13%
    )

    oof_label = (oof >= thr).astype(int)
    acc = accuracy_score(y_true, oof_label)
    f1 = f1_score(y_true, oof_label)
    pre = precision_score(y_true, oof_label)
    rec = recall_score(y_true, oof_label)
    score = 0.7 * acc + 0.3 * f1

    print(f"[OOF] Acc={acc:.5f}  F1={f1:.5f}  P={pre:.5f}  R={rec:.5f}  Thr={thr:.3f} Score: {score:.5f}")

    sub = pd.DataFrame({
        ID_COL: test[ID_COL].values,
        TARGET: (pred >= thr).astype(int)
    })
    sub.to_csv("submission.csv", index=False)
    print("[OK] 已生成 submission.csv")

    # 保存特征重要性
    if fi_df is not None and not fi_df.empty:
        fi_agg = fi_df.groupby("feature", as_index=False)["importance"].mean().sort_values("importance",ascending=False)
        fi_agg.to_csv("feature_importance.csv", index=False)
        print("[OK] 已生成 feature_importance.csv（gain 平均值）")


if __name__ == "__main__":
    main()