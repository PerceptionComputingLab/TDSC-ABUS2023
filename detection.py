#This code is based on https://github.com/M3DV/MELA-Challenge
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_KEY_FP = (0.125, 0.25, 0.5, 0.75, 1, 2, 4, 8)


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
    precision : float
        precision for this threshold.
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
    precision = (total_tp + EPS) / (total_tp + total_fp + EPS) 
    recall = (total_tp + EPS) / (total_gt + EPS)

    return precision, recall

def compute_average_precision(recall, precision):
    """
    Calculate mAP.

    Parameters
    recall: List of recall
    precision: list of precision

    Returns
    -------
    ap: Average precision
    """
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] =max(precision[i], precision[i+1])
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices-1]) * precision[indices])
    return ap

def froc(df_list, num_gts, iou_thresh=0.75):
    """
    Calculate the FROC curve.

    Parameters
    df_list : list of pandas.DataFrame
        List of prediction metrics.
    num_gts : list of int
        List of number of GT in each volume.
    iou_thresh : float
        The IoU threshold of predictions being considered as "hit".


    Returns
    -------
    precision_recall : List of recall and precision
    """
    precision_recall = [_froc_single_thresh(df_list, num_gts, p_thresh, iou_thresh)
                 for p_thresh in np.arange(0, 1, 0.005)]  

    return precision_recall



def score(gt_csv_path, pred_csv_path):
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
    map : Mean average precision.
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
    precision_recall = np.array(froc(det_results, num_gts))
    map = compute_average_precision(precision_recall[:,0],precision_recall[:,1])
    # print("precision\n",precision_recall[:,0])
    # print("recal\n",precision_recall[:,1])
    # print(map)

    return map

if __name__ == "__main__":
    eval_results = evaluate_map("gt_path", "pred_path")


