from dataclasses import dataclass

@dataclass
class POPEMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    yes_ratio: float
    total: int

    def __str__(self):
        return (
            f"Accuracy:  {self.accuracy:.2f}%\n"
            f"Precision: {self.precision:.2f}%\n"
            f"Recall:    {self.recall:.2f}%\n"
            f"F1 Score:  {self.f1:.2f}%\n"
            f"Yes Ratio: {self.yes_ratio:.2f}%\n"
            f"Total:     {self.total}"
        )


def compute_metrics(predictions: list, ground_truths: list) -> POPEMetrics:
    """
    Args:
        predictions:  list of "yes"/"no" strings (model output)
        ground_truths: list of "yes"/"no" strings (ground truth)

    Returns:
        POPEMetrics dataclass
    """
    assert len(predictions) == len(ground_truths), \
        "predictions and ground_truths must have the same length"

    total = len(predictions)

    # counts
    tp = 0   # predicted yes, answer yes
    fp = 0   # predicted yes, answer no
    tn = 0   # predicted no,  answer no
    fn = 0   # predicted no,  answer yes
    yes_count = 0

    for pred, gt in zip(predictions, ground_truths):
        pred = pred.strip().lower()
        gt = gt.strip().lower()

        if pred == "yes":
            yes_count += 1

        if pred == "yes" and gt == "yes":
            tp += 1
        elif pred == "yes" and gt == "no":
            fp += 1
        elif pred == "no" and gt == "no":
            tn += 1
        elif pred == "no" and gt == "yes":
            fn += 1

    accuracy  = 100.0 * (tp + tn) / total
    precision = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    yes_ratio = 100.0 * yes_count / total

    return POPEMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        yes_ratio=yes_ratio,
        total=total
    )