from argparse import ArgumentParser
from loguru import logger
import pandas as pd
import json
import os

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Evaluate submission.csv"
    )
    parser.add_argument(
        "--submission", type=str, required=True, help="Path to submission.csv file"
    )
    parser.add_argument(
        "--groundtruth", type=str, required=True, help="Path to ground_truth.json file"
    )
    parser.add_argument(
        "--see-gt-csv",
        action="store_true",
        default=False,
        help="Convert ground truth JSON to CSV format",
    )
    parser.add_argument(
        "--see-log-file",
        action="store_true",
        default=False,
        help="Save logs to a file instead of console",
    )
    args = parser.parse_args()

    # Configure logger
    if args.see_log_file:
        os.makedirs("logs", exist_ok=True)
        if os.path.exists("logs/evaluation_log.txt"):
            os.remove("logs/evaluation_log.txt")
        logger.add(
            "logs/evaluation_log.txt",
            format="{message}",
            level="INFO",
            rotation="10 MB",
        )

    # Load submission
    submission_df = pd.read_csv(args.submission)

    # Load ground truth
    with open(args.groundtruth, "r", encoding="utf-8") as f:
        ground_truth_json = json.load(f)

    # Prepare ground truth DataFrame
    ground_truth_file_ids = []
    ground_truth_answers = []
    for item in ground_truth_json["data"]:
        ground_truth_file_ids.append(item["id"])
        ground_truth_answers.append(item["answer"].split(".")[0])

    if args.see_gt_csv:
        logger.info("Converting ground truth JSON to CSV format...")
        gt_df = pd.DataFrame(
            {"id": ground_truth_file_ids, "answer": ground_truth_answers}
        )
        gt_df.to_csv("output/ground_truth.csv", index=False, encoding="utf-8")
        logger.info("Ground truth CSV file created: ground_truth.csv")

    correct_count = 0
    total_count = len(ground_truth_file_ids)
    for index, row in submission_df.iterrows():
        file_id = row["id"]
        predicted_answer = row["answer"]
        # if time column exists
        time_taken = row["time"] if "time" in row else None

        if file_id in ground_truth_file_ids:
            gt_index = ground_truth_file_ids.index(file_id)
            true_answer = ground_truth_answers[gt_index]

            if predicted_answer == true_answer and (
                time_taken is None or time_taken <= 30
            ):
                logger.info(f"{file_id}: Correct")
                correct_count += 1
            elif time_taken is not None and time_taken > 30:
                logger.warning(f"{file_id}: Time limit exceeded (Time: {time_taken}s)")
            else:
                logger.warning(
                    f"{file_id}: Incorrect (Predicted: {predicted_answer}, True: {true_answer})"
                )
        else:
            logger.warning(f"{file_id}: Not found in ground truth")

    logger.info("=====================================================")
    logger.info("Final Results:")
    logger.info(f"Correct: {correct_count}, Total: {total_count}")
    accuracy = correct_count / total_count if total_count > 0 else 0
    logger.info(f"Accuracy: {accuracy:.4f}")

    logger.info("Evaluation completed.")

#   python .\evaluate.py --submission submit.csv --groundtruth train.json --see-gt-csv --see-log-file
