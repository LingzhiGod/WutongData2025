#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型结果对比工具
------------------------------------
用法：
python compare_models.py submission_A.csv submission_B.csv [--label train_label.csv]

功能：
1. 对比两个模型预测结果差异
2. 输出总体差异率、F1/Acc 对比
3. 若提供真实标签文件，可计算真实性能差异
4. 输出详细报告 compare_report.txt
"""

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def load_csv(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [c.strip().lower() for c in df.columns]
    if "user_id" not in df.columns or "is_positive" not in df.columns:
        raise ValueError(f"{fp} 格式错误，应包含 user_id,is_positive 列")
    return df.sort_values("user_id").reset_index(drop=True)

def main():
    if len(sys.argv) < 3:
        print("用法: python compare_models.py result_A.csv result_B.csv [--label train_label.csv]")
        sys.exit(1)

    file_a = sys.argv[1]
    file_b = sys.argv[2]
    label_fp = None
    if len(sys.argv) > 3 and "--label" in sys.argv:
        label_fp = sys.argv[sys.argv.index("--label") + 1]

    dfA = load_csv(file_a)
    dfB = load_csv(file_b)

    if not np.array_equal(dfA["user_id"].values, dfB["user_id"].values):
        print("⚠️ 两个结果文件的 user_id 顺序不一致，将自动对齐。")
        dfB = dfB.set_index("user_id").reindex(dfA["user_id"]).reset_index()

    predA = dfA["is_positive"].astype(int).values
    predB = dfB["is_positive"].astype(int).values

    diff_mask = predA != predB
    diff_count = diff_mask.sum()
    diff_rate = diff_count / len(predA)

    print(f"📊 样本总数: {len(predA)}")
    print(f"✅ 两次预测相同样本数: {len(predA) - diff_count}")
    print(f"⚠️ 不同样本数: {diff_count} ({diff_rate:.4%})")

    # 若有标签文件
    metrics_text = ""
    if label_fp:
        dfY = load_csv(label_fp)
        dfY = dfY.set_index("user_id").reindex(dfA["user_id"]).reset_index()
        y_true = dfY["is_positive"].astype(int).values

        def calc_metrics(y_true, y_pred):
            return dict(
                acc = accuracy_score(y_true, y_pred),
                f1 = f1_score(y_true, y_pred),
                pre = precision_score(y_true, y_pred),
                rec = recall_score(y_true, y_pred)
            )

        mA = calc_metrics(y_true, predA)
        mB = calc_metrics(y_true, predB)

        metrics_text = (
            f"\n📈 结果指标对比：\n"
            f"  模型A  Acc={mA['acc']:.5f}  F1={mA['f1']:.5f}  P={mA['pre']:.5f}  R={mA['rec']:.5f}\n"
            f"  模型B  Acc={mB['acc']:.5f}  F1={mB['f1']:.5f}  P={mB['pre']:.5f}  R={mB['rec']:.5f}\n"
            f"  ΔF1={mB['f1']-mA['f1']:+.5f}  ΔAcc={mB['acc']-mA['acc']:+.5f}\n"
        )
        print(metrics_text)

    # 差异样本列表
    diff_users = dfA.loc[diff_mask, "user_id"].tolist()
    diff_df = dfA.loc[diff_mask, ["user_id"]].copy()
    diff_df["pred_A"] = predA[diff_mask]
    diff_df["pred_B"] = predB[diff_mask]

    # 保存报告
    report = f"""
模型结果对比报告
==============================
文件A: {file_a}
文件B: {file_b}

样本总数: {len(predA)}
预测不一致数: {diff_count} ({diff_rate:.4%})

{metrics_text if metrics_text else '(无真实标签, 仅对比预测差异)'}
前10个预测不同的样本ID:
{diff_users[:10]}

详细不同样本已保存至 diff_samples.csv
"""
    with open("compare_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    diff_df.to_csv("diff_samples.csv", index=False)

    print("✅ 已生成 compare_report.txt 与 diff_samples.csv")

if __name__ == "__main__":
    main()
