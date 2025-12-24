import os
# Force TensorFlow to use CPU only (must be set before importing tensorflow)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import numpy as np
from bleurt import score as bleurt_score
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_INPUT_PATH = PROJECT_ROOT / "HAD/outputs/test_with_answers.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "HAD/outputs/test_bleurt_evaluation.json"

def load_bleurt_scorer(checkpoint="BLEURT-20"):
    """Load BLEURT scorer with specified checkpoint"""
    print(f"Loading BLEURT model ({checkpoint})...")
    scorer = bleurt_score.BleurtScorer(checkpoint)
    return scorer

def calculate_bleurt(predictions, references, scorer, batch_size=32):
    """
    Calculate BLEURT scores for prediction-reference pairs
    """
    print("Calculating BLEURT scores...")
    all_scores = []
    
    # Process in batches for efficiency
    for i in tqdm(range(0, len(predictions), batch_size)):
        batch_preds = predictions[i:i + batch_size]
        batch_refs = references[i:i + batch_size]
        scores = scorer.score(references=batch_refs, candidates=batch_preds)
        all_scores.extend(scores)
    
    return all_scores

def aggregate_scores(scores):
    """Calculate mean and std for BLEURT scores"""
    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores))
    }

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH, help="Path to input JSON file with model answers")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to output JSON file for evaluation results")
    args = parser.parse_args()
    input_json_path = args.input
    output_json_path = args.output
    
    print(f"Loading results from {input_json_path}")
    with open(input_json_path, "r") as f:
        test_data = json.load(f)
    
    print(f"Total samples: {len(test_data)}")
    
    # Extract predictions and references
    predictions = []
    references = []
    valid_indices = []
    
    for idx, item in enumerate(test_data):
        model_answer = item.get("model_answer", "")
        ground_truth = item.get("QA", {}).get("a", "")
        
        # Skip if error or empty
        if model_answer.startswith("ERROR") or not model_answer or not ground_truth:
            continue
        
        predictions.append(model_answer)
        references.append(ground_truth)
        valid_indices.append(idx)
    
    print(f"Valid samples for evaluation: {len(predictions)}")
    
    # Load BLEURT scorer
    scorer = load_bleurt_scorer()
    
    # Calculate BLEURT scores
    bleurt_scores = calculate_bleurt(predictions, references, scorer)
    
    # Aggregate scores
    aggregated = aggregate_scores(bleurt_scores)
    
    # Print results
    print("\n" + "="*50)
    print("BLEURT EVALUATION RESULTS")
    print("="*50)
    print(f"BLEURT Score: {aggregated['mean']:.4f} ± {aggregated['std']:.4f}")
    print(f"Min: {aggregated['min']:.4f}, Max: {aggregated['max']:.4f}")
    print("="*50)
    
    # Save results
    output_data = {
        'aggregated_metrics': {
            'bleurt': aggregated
        },
        'per_sample_scores': bleurt_scores,
        'num_samples': len(predictions)
    }
    
    print(f"\nSaving evaluation results to {output_json_path}")
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()