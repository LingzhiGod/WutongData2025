import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

# === 假设你的训练脚本名为 core.py，且与本脚本在同一目录 ===
from core import (
    load_data,
    build_features,
    compute_scale,
    lgbm_default_params,
    RANDOM_STATE,
    TARGET,
    pick_best_threshold_by_score,
)

N_TRIALS = 100

# =========================================================
# 预先构建一次特征，避免在每个 trial 里重复做特征工程
# =========================================================
print("[Optuna] Loading data & building features (once)...")
train, _ = load_data()
train_fe = build_features(train, is_train=True)

USING_COLS = train_fe["__used_cols__"].iloc[0].split(",")
CAT_LE_COLS = train_fe["__cat_le_cols__"].iloc[0].split(",")

X = train_fe[USING_COLS]
y = train[TARGET].astype(int).values
print(f"[Optuna] X shape = {X.shape}, y_pos_rate = {y.mean():.5f}")


def objective(trial: optuna.trial.Trial) -> float:
    """单次 trial 的目标函数：返回平台 Score（越大越好）"""

    params = lgbm_default_params.copy()

    # ---------- 搜索的超参数 ----------
    params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.08, log=True)
    params["num_leaves"] = trial.suggest_int("num_leaves", 16, 127)
    params["max_depth"] = trial.suggest_int("max_depth", 4, 9)
    params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 50, 400)
    params["min_split_gain"] = trial.suggest_float("min_split_gain", 0.0, 1.0)

    params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.6, 0.95)
    params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.6, 0.95)
    params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 5)

    params["lambda_l1"] = trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True)
    params["lambda_l2"] = trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True)

    # 类别不平衡：用 scale_pos_weight 替换 is_unbalance
    params.pop("is_unbalance", None)
    params["scale_pos_weight"] = compute_scale(y)

    # ---------- 这里把 num_boost_round & early_stopping_rounds 也纳入调参 ----------
    num_boost_round = trial.suggest_int("num_boost_round", 300, 2000)
    early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 50, 300)

    # ---------- 5 折交叉验证 ----------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(y), dtype=float)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]

        dtrain = lgb.Dataset(
            X_trn,
            label=y_trn,
            categorical_feature=CAT_LE_COLS,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=CAT_LE_COLS,
            reference=dtrain,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                # early_stopping 回调
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                # 不在每轮打印日志
                lgb.log_evaluation(0),
            ],
        )

        oof_pred[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

    # ---------- 用你自己的打分逻辑做阈值搜索 ----------
    best_thr, best_score, (acc, f1, p, r) = pick_best_threshold_by_score(
        y_true=y,
        prob=oof_pred,
        step=0.005,  # 与你原来的保持一致
    )

    # 把一些信息挂在 trial 上，方便后面输出
    trial.set_user_attr("thr", float(best_thr))
    trial.set_user_attr("acc", float(acc))
    trial.set_user_attr("f1", float(f1))
    trial.set_user_attr("precision", float(p))
    trial.set_user_attr("recall", float(r))

    return float(best_score)  # Optuna 将会最大化这个 Score


def main():
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="lgbm_optuna_tuning",
    )

    # n_trials 可按实际情况调整
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_trial = study.best_trial
    print("\n========== Optuna Tuning Finished ==========")
    print(f"Best Score (0.7*Acc+0.3*F1): {best_trial.value:.5f}")
    print("Best params:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")

    print("Best threshold:", best_trial.user_attrs.get("thr"))
    print("Best Acc:", best_trial.user_attrs.get("acc"))
    print("Best F1:", best_trial.user_attrs.get("f1"))
    print("Best Precision:", best_trial.user_attrs.get("precision"))
    print("Best Recall:", best_trial.user_attrs.get("recall"))

    # 保存为 core.py 中 load_best_params 可读取的格式
    best_params = best_trial.params.copy()
    # core.py 里会把这一列当作 best_f1 打印，可以存真正的 F1（或 Score，随你）
    best_params["best_f1"] = best_trial.user_attrs.get("f1", None)

    df = pd.DataFrame([best_params])
    df.to_csv("optuna_best_params.csv", index=False)
    print("[OK] Saved best params to optuna_best_params.csv")


if __name__ == "__main__":
    main()
