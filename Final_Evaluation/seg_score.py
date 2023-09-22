import argparse
import csv
import os

import SimpleITK
from alive_progress import alive_it

parser = argparse.ArgumentParser()
parser.add_argument("team_name", type=str, help="The team name")


def seg_score_per_case(gt_file, pred_file):
    # Calculate Dice HD score per case

    # Load the images for this case
    gt = SimpleITK.ReadImage(gt_file)
    pred = SimpleITK.ReadImage(pred_file)

    # Cast to the same type
    caster = SimpleITK.CastImageFilter()
    caster.SetOutputPixelType(SimpleITK.sitkUInt8)
    caster.SetNumberOfThreads(1)
    gt = caster.Execute(gt)
    pred = caster.Execute(pred)

    # Score the case
    overlap_measures = SimpleITK.LabelOverlapMeasuresImageFilter()
    overlap_measures.SetNumberOfThreads(1)
    overlap_measures.Execute(gt, pred)
    dice = overlap_measures.GetDiceCoefficient()
    try:
        hausdorff_filter = SimpleITK.HausdorffDistanceImageFilter()
        hausdorff_filter.Execute(gt, pred)
    except Exception:
        HD = float("inf")
    else:
        HD = hausdorff_filter.GetHausdorffDistance()

    return {
        "DiceCoefficient": dice,
        "HDCoefficient": HD,
        "score": dice - HD,
    }


def calculate_seg_score_all(gt_path, pred_path, team):
    # Iterate over all cases in the test set and calculate the score

    def sort_filename(name):
        key = int(name.split("_")[1].split(".", 1)[0])
        return key

    # Get the list of cases
    cases = os.listdir(gt_path)
    cases.sort(key=sort_filename)

    # Get the predict file format
    pred_cases = os.listdir(pred_path)
    pred_cases.sort(key=sort_filename)

    assert len(cases) == len(pred_cases)

    # Calculate the score for each case
    scores = []
    csvfile = open(f"{team}/seg_score.csv", "w", encoding="utf-8")
    writer = csv.DictWriter(
        csvfile, fieldnames=[
            "case", "DiceCoefficient", "HDCoefficient", "score"]
    )
    writer.writeheader()
    for gt, pred in alive_it(zip(cases, pred_cases), total=len(pred_cases), title=team):
        res = seg_score_per_case(
            os.path.join(gt_path, gt), os.path.join(pred_path, pred)
        )
        writer.writerow({**res, "case": gt})
        scores.append(res)

    # Calculate the mean score
    mean_dice = sum([s["DiceCoefficient"] for s in scores]) / len(scores)
    mean_HD = sum([s["HDCoefficient"] for s in scores]) / len(scores)
    mean_score = sum([s["score"] for s in scores]) / len(scores)
    writer.writerow(
        {
            "case": team,
            "DiceCoefficient": mean_dice,
            "HDCoefficient": mean_HD,
            "score": mean_score,
        }
    )
    csvfile.close()
    return {"dice": mean_dice, "HD": mean_HD, "score": mean_score}


if __name__ == "__main__":
    args = parser.parse_args()
    team = args.team_name
    if len(os.listdir(f"{team}/predict/Segmentation")) != 0:
        seg_score = calculate_seg_score_all(
            "./Test/MASK",
            os.path.join(team, "predict", "Segmentation"),
            team,
        )
        print(f"{team} seg score: {seg_score}")
