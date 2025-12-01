"""
Evaluate similarity between two CSV prediction/label files.

Each file format:
id,filename,answer
1,xxx.mp4,2
...

Rules:
- Match by `id` (order doesn't matter).
- Compare `answer` values.
- Report:
    * number of correct answers
    * number of wrong answers
    * missing ids in either file
- Export wrong answers to a CSV.

Usage:
  python evaluate.py --gt_path ground_truth.csv --pred_path predictions.csv
Optional:
  python evaluate.py --gt_path gt.csv --pred_path pred.csv --wrong_out wrong.csv
"""

import argparse
import csv
from pathlib import Path


def read_csv_as_dict(path):
    """
    Returns dict: id(int) -> {"filename": str, "answer": str}
    """
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "filename", "answer"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must have columns: {sorted(required)}. "
                f"Got: {reader.fieldnames}"
            )

        for row_num, row in enumerate(reader, start=2):
            try:
                id_int = int(row["id"])
            except Exception:
                raise ValueError(f"Invalid id on line {row_num} in {path}: {row['id']}")

            data[id_int] = {
                "filename": row["filename"],
                "answer": str(row["answer"]).strip(),
            }
    return data


def evaluate(gt_path, pred_path, wrong_out_path=None):
    gt = read_csv_as_dict(gt_path)
    pred = read_csv_as_dict(pred_path)

    gt_ids = set(gt.keys())
    pred_ids = set(pred.keys())
    common_ids = gt_ids & pred_ids
    missing_in_pred = sorted(gt_ids - pred_ids)
    missing_in_gt = sorted(pred_ids - gt_ids)

    correct = 0
    wrong = 0
    wrong_rows = []

    for id_ in sorted(common_ids):
        gt_ans = gt[id_]["answer"]
        pred_ans = pred[id_]["answer"]

        if pred_ans == gt_ans:
            correct += 1
        else:
            wrong += 1
            wrong_rows.append({
                "id": id_,
                "filename": gt[id_]["filename"] or pred[id_]["filename"],
                "gt_answer": gt_ans,
                "pred_answer": pred_ans,
            })

    total_compared = correct + wrong
    accuracy = (correct / total_compared) if total_compared else 0.0

    # default wrong_out_path
    if wrong_out_path is None:
        wrong_out_path = Path(pred_path).with_suffix("").as_posix() + "_wrong.csv"

    # write wrong answers
    with open(wrong_out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "filename", "gt_answer", "pred_answer"]
        )
        writer.writeheader()
        writer.writerows(wrong_rows)

    # print report
    print("=== Evaluation Report ===")
    print(f"GT file:   {gt_path}")
    print(f"PRED file: {pred_path}")
    print(f"Total GT ids:    {len(gt_ids)}")
    print(f"Total PRED ids:  {len(pred_ids)}")
    print(f"Compared ids:    {len(common_ids)}")
    print(f"Correct:         {correct}")
    print(f"Wrong:           {wrong}")
    print(f"Accuracy:        {accuracy * 100:.4f}%")

    if missing_in_pred:
        print(f"\nMissing in PRED ({len(missing_in_pred)} ids):")
        print(missing_in_pred[:50], "..." if len(missing_in_pred) > 50 else "")

    if missing_in_gt:
        print(f"\nExtra ids in PRED not in GT ({len(missing_in_gt)} ids):")
        print(missing_in_gt[:50], "..." if len(missing_in_gt) > 50 else "")

    print(f"\nWrong answers written to: {wrong_out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", required=True, help="Ground truth CSV path")
    parser.add_argument("--pred_path", required=True, help="Prediction CSV path")
    parser.add_argument("--wrong_path", default=None, help="Optional path to save wrong answers CSV")
    args = parser.parse_args()

    if not args.gt_path.endswith(".csv"):
        raise ValueError("Ground truth file must have a .csv extension.")

    if not args.pred_path.endswith(".csv"):
        raise ValueError("Prediction file must have a .csv extension.")

    if args.wrong_path is not None and not args.wrong_path.endswith(".csv"):
        raise ValueError("Wrong answers file must have a .csv extension.")

    evaluate(args.gt_path, args.pred_path, args.wrong_path)

if __name__ == "__main__":
    main()
