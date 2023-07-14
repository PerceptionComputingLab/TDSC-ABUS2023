import SimpleITK

def score_case(gt_path, pred_path):
    # Load the images for this case
    gt = SimpleITK.ReadImage(gt_path)
    pred = SimpleITK.ReadImage(pred_path)

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
    hausdorff_filter = SimpleITK.HausdorffDistanceImageFilter()
    hausdorff_filter.Execute(gt, pred)

    return {
            'DiceCoefficient': overlap_measures.GetDiceCoefficient(),
            'HDCoefficient': hausdorff_filter.GetHausdorffDistance(),
            'score': overlap_measures.GetDiceCoefficient()-hausdorff_filter.GetHausdorffDistance(),
        }

if __name__ == "__main__":
    """
    Evaluate a single case
    """
    eval_results = score_case("gt_path", "pred_path")