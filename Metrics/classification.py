from sklearn.metrics import accuracy_score,roc_auc_score

import pandas as pd

def score_aggregates(gt_csv_path, pred_csv_path):
    gt_info = pd.read_csv(gt_csv_path)
    gt_info = gt_info["class"]
    pred_info = pd.read_csv(pred_csv_path)
    pred_info = pred_info["class"]
    label_map = {'M': 1, 'B': 0}
    y_true = [label_map[label] for label in gt_info]
    y_pred = [label_map[label] for label in pred_info]
    return {
        "accuracy": accuracy_score(y_true,y_pred),
        "auc": roc_auc_score(y_true,y_pred),
        "score": (roc_auc_score(y_true,y_pred)+accuracy_score(y_true,y_pred))/2
    }



if __name__ == "__main__":
    eval_results = score_aggregates("gt_path", "pred_path")
