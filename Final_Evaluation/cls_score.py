import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("team_name", type=str, help="The team name")


def cls_score(gt_csv_path, pred_csv_path):
    # Calculate ACC AUC score per case

    gt_info = pd.read_csv(gt_csv_path)
    gt_info = gt_info["label"]
    pred_info = pd.read_csv(pred_csv_path)
    pred_info = pred_info.sort_values("case")
    pred_info = pred_info["prob"]
    label_map = {"M": 1, "B": 0}
    y_true = [label_map[label] for label in gt_info]
    y_pred = pred_info.apply(lambda x: 1 if x >= 0.5 else 0)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, pred_info)
    return {
        "accuracy": acc,
        "auc": auc,
        "score": (acc + auc) / 2,
    }


if __name__ == "__main__":
    args = parser.parse_args()
    team = args.team_name
    if len(os.listdir(f"{team}/predict/Classification")) != 0:
        score = cls_score(
            "./Test/labels.csv",
            os.path.join(
                team,
                "predict",
                "Classification",
                os.listdir(f"{team}/predict/Classification")[0],
            ),
        )
        print(f"{team} cls score: {score}")
