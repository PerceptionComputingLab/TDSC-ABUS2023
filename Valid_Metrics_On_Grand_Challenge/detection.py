#This code if from https://github.com/M3DV/MELA-Challenge/tree/main/MELA
# --*-- coding:utf-8 -*-
import cv2
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


DEFAULT_KEY_FP = (0.125, 0.25, 0.5, 1, 2, 4, 8)


def iou_3d(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two 3D bounding boxes.
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]
    (x1_c, y1_c, z1_c, w1, h1, d1) = bbox1
    (x2_c, y2_c, z2_c, w2, h2, d2) = bbox2

    x1_1 = x1_c - w1 / 2
    x1_2 = x1_c + w1 / 2
    y1_1 = y1_c - h1 / 2
    y1_2 = y1_c + h1 / 2
    z1_1 = z1_c - d1 / 2
    z1_2 = z1_c + d1 / 2

    x2_1 = x2_c - w2 / 2
    x2_2 = x2_c + w2 / 2
    y2_1 = y2_c - h2 / 2
    y2_2 = y2_c + h2 / 2
    z2_1 = z2_c - d2 / 2
    z2_2 = z2_c + d2 / 2

    # get the overlap rectangle
    overlap_x1 = max(x1_1, x2_1)
    overlap_y1 = max(y1_1, y2_1)
    overlap_z1 = max(z1_1, z2_1)
    overlap_x2 = min(x1_2, x2_2)
    overlap_y2 = min(y1_2, y2_2)
    overlap_z2 = min(z1_2, z2_2)

    # check if there is an overlap
    if overlap_x2 - overlap_x1 <= 0 or overlap_y2 - overlap_y1 <= 0 or overlap_z2 - overlap_z1 <= 0:
        return 0

    size_1 = (x1_2 - x1_1) * (y1_2 - y1_1) * (z1_2 - z1_1)
    size_2 = (x2_2 - x2_1) * (y2_2 - y2_1) * (z2_2 - z2_1)
    size_intersection = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1) * (overlap_z2 - overlap_z1)
    size_union = size_1 + size_2 - size_intersection
    return size_intersection / size_union

def _compile_pred_metrics(iou_matrix, gt_info, pred_info):
    """
    Compile prediction metrics into a Pandas DataFrame

    Parameters
    ----------
    iou_matrix : numpy.ndarray
        IoU array with shape of (n_pred, n_gt).
    gt_info : pandas.DataFrame
        DataFrame containing GT information.
    pred_info : pandas.DataFrame
        DataFrame containing prediction information.

    Returns
    -------
    pred_metrics : pandas.DataFrame
        A dataframe of prediction metrics.
    """
    # meanings of each column:
    # pred_label --  The index of prediction
    # max_iou -- The highest IoU this prediction has with any certain GT
    # hit_label -- The GT label with which this prediction has the highest IoU
    # prob -- The confidence prediction of this prediction
    # num_gt -- Total number of GT in this volume

    pred_metrics = pd.DataFrame(np.zeros((iou_matrix.shape[0], 3)),
                                columns=["pred_label", "max_iou", "hit_label"])
    pred_metrics["pred_label"] = np.arange(1, iou_matrix.shape[0] + 1)
    pred_metrics["max_iou"] = iou_matrix.max(axis=1)
    pred_metrics["hit_label"] = iou_matrix.argmax(axis=1) + 1

    pred_metrics["hit_label"] = pred_metrics.apply(lambda x: x["hit_label"] if x["max_iou"] > 0 else 0, axis=1)
    # fill in the detection confidence

    pred_metrics = pred_metrics.merge(
        pred_info[["label_id", "probability"]],
        how="left", left_on="pred_label", right_on="label_id")
    pred_metrics.rename({"probability": "prob"},
                        axis=1, inplace=True)
    pred_metrics.drop("label_id", axis=1, inplace=True)

    pred_metrics = pred_metrics.merge(gt_info["label_id"],
                                      how="left", left_on="hit_label", right_on="label_id")
    pred_metrics.drop("label_id", axis=1, inplace=True)
    pred_metrics["num_gt"] = iou_matrix.shape[1]

    return pred_metrics


def evaluate_single_prediction(gt_info, pred_info):
    """
    Evaluate a single prediction.

    Parameters
    ----------
    gt_info : pandas.DataFrame
        DataFrame containing GT information.
    pred_info : pandas.DataFrame
        DataFrame containing prediction information.

    Returns
    -------
    pred_metrics : pandas.DataFrame
        A dataframe of prediction metrics.
    num_gt : int
        Number of GT in this case.
    """
    # GT and prediction

    num_gt = len(gt_info)
    num_pred = len(pred_info)

    # if the prediction is empty, return empty pred_metrics
    if num_pred == 0:
        pred_metrics = pd.DataFrame()
        return pred_metrics, num_gt

    # if GT is empty
    if num_gt == 0:
        pred_metrics = pd.DataFrame([
            {
                "pred_label": i,
                "max_iou": 0,
                "hit_label": 0,
                "gt_class": "FP",
                "num_gt": 0
            }
            for i in range(1, num_pred + 1)])
        pred_metrics = pred_metrics.merge(
            pred_info[["label_id", "probability"]],
            how="left", left_on="pred_label", right_on="label_id")
        pred_metrics.rename(
            {"probability": "prob"}, axis=1,
            inplace=True)
        pred_metrics.drop(["label_id"], axis=1, inplace=True)

        return pred_metrics, num_gt

    iou_matrix = np.zeros((num_gt, num_pred))

    # iterate through all gt and prediction of seriesuid and evaluate predictions
    for gt_idx in range(num_gt):
        # get gt bbox info
        gt_bbox_info = gt_info.iloc[gt_idx]
        gt_bbox = [gt_bbox_info['coordX'], gt_bbox_info['coordY'], gt_bbox_info['coordZ'],
                   gt_bbox_info['x_length'], gt_bbox_info['y_length'], gt_bbox_info['z_length']]

        for pred_idx in range(num_pred):
            # get prediction bbox info
            pred_bbox_info = pred_info.iloc[pred_idx]
            pred_bbox = [pred_bbox_info['coordX'], pred_bbox_info['coordY'], pred_bbox_info['coordZ'],
                         pred_bbox_info['x_length'], pred_bbox_info['y_length'], pred_bbox_info['z_length']]
            # get iou of gt and pred bboxes
            gt_pred_iou = iou_3d(gt_bbox, pred_bbox)
            iou_matrix[gt_idx, pred_idx] = gt_pred_iou

        # get corresponding GT index, pred index and union index

    iou_matrix = iou_matrix.T
    pred_metrics = _compile_pred_metrics(iou_matrix, gt_info, pred_info)

    return pred_metrics, num_gt


def _froc_single_thresh(df_list, num_gts, p_thresh, iou_thresh):
    """
    Calculate the FROC for a single confidence threshold.

    Parameters
    ----------
    df_list : list of pandas.DataFrame
        List of Pandas DataFrame of prediction metrics.
    num_gts : list of int
        List of number of GT in each volume.
    p_thresh : float
        The probability threshold of positive predictions.
    iou_thresh : float
        The IoU threshold of predictions being considered as "hit".

    Returns
    -------
    fp : float
        False positives per scan for this threshold.
    recall : float
        Recall rate for this threshold.
    """
    EPS = 1e-8

    total_gt = sum(num_gts)
    # collect all predictions above the probability threshold
    df_pos_pred = [df.loc[df["prob"] >= p_thresh] for df in df_list
                   if len(df) > 0]

    # calculate total true positives
    total_tp = sum([len(df.loc[df["max_iou"] > iou_thresh, "hit_label"] \
                        .unique()) for df in df_pos_pred])

    # calculate total false positives
    total_fp = sum([len(df) - len(df.loc[df["max_iou"] > iou_thresh])
                    for df in df_pos_pred])

    fp = (total_fp + EPS) / (len(df_list) + EPS)  # average fp in every sample
    recall = (total_tp + EPS) / (total_gt + EPS)

    return fp, recall


def _interpolate_recall_at_fp(fp_recall, key_fp):
    """
    Calculate recall at key_fp using interpolation.

    Parameters
    ----------
    fp_recall : pandas.DataFrame
        DataFrame of FP and recall.
    key_fp : float
        Key FP threshold at which the recall will be calculated.

    Returns
    -------
    recall_at_fp : float
        Recall at key_fp.
    """
    # get fp/recall interpolation points
    fp_recall_less_fp = fp_recall.loc[fp_recall.fp <= key_fp]
    fp_recall_more_fp = fp_recall.loc[fp_recall.fp >= key_fp]

    # if key_fp < min_fp, recall = 0
    if len(fp_recall_less_fp) == 0:
        return 0

    # if key_fp > max_fp, recall = max_recall
    if len(fp_recall_more_fp) == 0:
        return fp_recall.recall.max()

    fp_0 = fp_recall_less_fp["fp"].values[-1]
    fp_1 = fp_recall_more_fp["fp"].values[0]
    recall_0 = fp_recall_less_fp["recall"].values[-1]
    recall_1 = fp_recall_more_fp["recall"].values[0]
    recall_at_fp = recall_0 + (recall_1 - recall_0) \
                   * ((key_fp - fp_0) / (fp_1 - fp_0 + 1e-8))

    return recall_at_fp


def _get_key_recall(fp, recall, key_fp_list):
    """
    Calculate recall at a series of FP threshold.

    Parameters
    ----------
    fp : list of float
        List of FP at different probability thresholds.
    recall : list of float
        List of recall at different probability thresholds.
    key_fp_list : list of float
        List of key FP values.

    Returns
    -------
    key_recall : list of float
        List of key recall at each key FP.
    """
    fp_recall = pd.DataFrame({"fp": fp, "recall": recall}).sort_values("fp")
    key_recall = [_interpolate_recall_at_fp(fp_recall, key_fp)
                  for key_fp in key_fp_list]

    return key_recall


def froc(df_list, num_gts, iou_thresh=0.3, key_fp=DEFAULT_KEY_FP):
    """
    Calculate the FROC curve.

    Parameters
    df_list : list of pandas.DataFrame
        List of prediction metrics.
    num_gts : list of int
        List of number of GT in each volume.
    iou_thresh : float
        The IoU threshold of predictions being considered as "hit".
    key_fp : tuple of float
        The key false positive per scan used in evaluating the sensitivity
        of the model.

    Returns
    -------
    fp : list of float
        List of false positives per scan at different probability thresholds.
    recall : list of float
        List of recall at different probability thresholds.
    key_recall : list of float
        List of key recall corresponding to key FPs.
    avg_recall : float
        Average recall at key FPs. This is the evaluation metric we use
        in the detection track.
    """
    fp_recall = [_froc_single_thresh(df_list, num_gts, p_thresh, iou_thresh)
                 for p_thresh in np.arange(0, 1, 0.005)]  
    fp = [x[0] for x in fp_recall]
    recall = [x[1] for x in fp_recall]
    key_recall = _get_key_recall(fp, recall, key_fp)
    avg_recall = np.mean(key_recall)

    return fp, recall, key_recall, avg_recall


def plot_froc(fp, recall):
    """
    Plot the FROC curve.

    Parameters
    ----------
    fp : list of float
        List of false positive per scans at different confidence thresholds.
    recall : list of float
        List of recall at different confidence thresholds.
    """
    _, ax = plt.subplots()
    ax.plot(fp, recall)
    ax.set_title("FROC")
    plt.savefig("froc.jpg")


def evaluate(gt_csv_path, pred_csv_path):
    """
    Evaluate predictions against the ground-truth.

    Parameters
    ----------
    gt_csv_path : str
        The ground-truth csv.
    pred_csv_path : str
        The prediction csv.

    Returns
    -------
    eval_results : dict
        Dictionary containing detection results.
    """
    # GT and prediction information
    gt_info = pd.read_csv(gt_csv_path)
    pred_info = pd.read_csv(pred_csv_path)
    gt_pids = list(gt_info["public_id"].unique())
    pred_pids = list(pred_info["public_id"].unique())

    # GT and prediction directory sanity check
    for i in pred_pids:
        assert i in gt_pids, \
            "Unmatched seriesuid (not included in test set)."

    eval_results = []
    progress = tqdm(total=len(gt_pids))
    for pid in gt_pids:
        # get GT array and information
        cur_gt_info = gt_info.loc[gt_info.public_id == pid] \
            .reset_index(drop=True)
        cur_gt_info['label_id'] = np.arange(1, len(cur_gt_info) + 1)

        # get prediction array and information
        cur_pred_info = pred_info.loc[pred_info.public_id == pid] \
            .reset_index(drop=True)
        cur_pred_info['label_id'] = np.arange(1, len(cur_pred_info) + 1)

        # perform evaluation
        eval_results.append(evaluate_single_prediction(cur_gt_info, cur_pred_info))

        progress.update(1)

    progress.close()

    # detection results
    det_results = [x[0] for x in eval_results]
    num_gts = [x[1] for x in eval_results]

    # calculate the detection FROC
    fp, recall, key_recall, avg_recall = froc(det_results, num_gts)

    eval_results = {
        "detection": {
            "fp": fp,
            "recall": recall,
            "key_recall": key_recall,
            "average_recall": avg_recall,
            "max_recall": max(recall),
            "average_fp_at_max_recall": max(fp),
        }
    }

    return eval_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--clf", default="True")
    args = parser.parse_args()
    eval_results = evaluate(args.gt_dir, args.pred_dir)

    # detection metrics
    print("\nDetection metrics")
    print("=" * 64)
    print("Recall at key FP")
    froc_recall = pd.DataFrame(np.array(eval_results["detection"]["key_recall"]) \
                               .reshape(1, -1), index=["Recall"],
                               columns=[f"FP={str(x)}" for x in DEFAULT_KEY_FP])
    print(froc_recall)
    print("Average recall: {:.4f}".format(
        eval_results["detection"]["average_recall"]))
    print("Maximum recall: {:.4f}".format(
        eval_results["detection"]["max_recall"]
    ))
    print("Average FP per scan at maximum recall: {:.4f}".format(
        eval_results["detection"]["average_fp_at_max_recall"]
    ))

    # plot/print FROC curve
    print("FPR, Recall in FROC")
    for fp, recall in zip(reversed(eval_results["detection"]["fp"]),
                          reversed(eval_results["detection"]["recall"])):
        print(f"({fp:.8f}, {recall:.8f})")
    plot_froc(eval_results["detection"]["fp"], eval_results["detection"]["recall"])
