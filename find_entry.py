import json
import numpy as np

# Load the evaluation results
eval_path = "/datastore/clc_hcmus/ZaAIC/CS412-CV-FinalProject/HAD/questions/test_bleurt_evaluation.json"
answers_path = "/datastore/clc_hcmus/ZaAIC/CS412-CV-FinalProject/HAD/questions/test_with_answers.json"
output_path = "/datastore/clc_hcmus/ZaAIC/CS412-CV-FinalProject/HAD/questions/test_best_worst_5percent.json"

with open(eval_path, "r") as f:
    eval_data = json.load(f)

with open(answers_path, "r") as f:
    answers_data = json.load(f)

# Get per-sample scores
scores = eval_data["per_sample_scores"]
num_samples = len(scores)

# Calculate 5% thresholds
top_5_percent_count = max(1, int(num_samples * 0.05))
bottom_5_percent_count = max(1, int(num_samples * 0.05))

# Get indices sorted by score
sorted_indices = np.argsort(scores)

# Get top 5% (highest scores) and bottom 5% (lowest scores)
worst_indices = sorted_indices[:bottom_5_percent_count].tolist()
best_indices = sorted_indices[-top_5_percent_count:].tolist()[::-1]  # Reverse to have highest first

# Prepare output data
def extract_sample_info(idx, score):
    sample = answers_data[idx]
    return {
        "index": idx,
        "bleurt_score": score,
        "video_id": sample.get("video_id", "N/A"),
        "question": sample.get("question", "N/A"),
        "ground_truth": sample.get("QA", {}).get("a", "N/A"),
        "model_answer": sample.get("model_answer", "N/A")
    }

best_samples = [extract_sample_info(idx, scores[idx]) for idx in best_indices]
worst_samples = [extract_sample_info(idx, scores[idx]) for idx in worst_indices]

# Create output
output_data = {
    "statistics": {
        "total_samples": num_samples,
        "top_5_percent_count": top_5_percent_count,
        "bottom_5_percent_count": bottom_5_percent_count,
        "best_score_threshold": scores[sorted_indices[-top_5_percent_count]],
        "worst_score_threshold": scores[sorted_indices[bottom_5_percent_count - 1]],
        "overall_mean": np.mean(scores),
        "overall_std": np.std(scores)
    },
    "top_5_percent_best": best_samples,
    "top_5_percent_worst": worst_samples
}

# Save to file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

# Print summary
print("=" * 80)
print("STATISTICS")
print("=" * 80)
print(f"Total samples: {num_samples}")
print(f"Top 5% count: {top_5_percent_count}")
print(f"Bottom 5% count: {bottom_5_percent_count}")
print(f"Overall mean: {np.mean(scores):.4f}")
print(f"Overall std: {np.std(scores):.4f}")
print(f"Best score threshold (top 5%): {output_data['statistics']['best_score_threshold']:.4f}")
print(f"Worst score threshold (bottom 5%): {output_data['statistics']['worst_score_threshold']:.4f}")

print("\n" + "=" * 80)
print("TOP 5% BEST SAMPLES")
print("=" * 80)
for i, sample in enumerate(best_samples[:5]):  # Print first 5
    print(f"\n--- Sample {i+1} (Index: {sample['index']}, Score: {sample['bleurt_score']:.4f}) ---")
    print(f"Video ID: {sample['video_id']}")
    print(f"Question: {sample['question'][:200]}...")
    print(f"Ground Truth: {sample['ground_truth'][:200]}...")
    print(f"Model Answer: {sample['model_answer'][:200]}...")

print("\n" + "=" * 80)
print("TOP 5% WORST SAMPLES")
print("=" * 80)
for i, sample in enumerate(worst_samples[:5]):  # Print first 5
    print(f"\n--- Sample {i+1} (Index: {sample['index']}, Score: {sample['bleurt_score']:.4f}) ---")
    print(f"Video ID: {sample['video_id']}")
    print(f"Question: {sample['question'][:200]}...")
    print(f"Ground Truth: {sample['ground_truth'][:200]}...")
    print(f"Model Answer: {sample['model_answer'][:200]}...")

print(f"\n{'=' * 80}")
print(f"Full results saved to: {output_path}")