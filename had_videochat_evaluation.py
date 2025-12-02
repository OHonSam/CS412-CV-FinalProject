import json
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def load_judge_model():
    """Load Qwen3-4B-Instruct model for LLM-as-a-judge evaluation"""
    print("Loading LLM judge model (Qwen3-4B-Instruct-2507)...")
    model_name = "Qwen/Qwen3-4B-Instruct-2507"  # Using Qwen2.5-3B as Qwen3-4B might not be available
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def llm_judge_score(question, prediction, reference, model, tokenizer):
    """
    Use LLM to judge the quality of prediction compared to reference
    Returns a score from 0 to 5
    """
    prompt = f"""You are an expert evaluator for video question answering tasks. Your job is to rate the quality of a predicted answer compared to a reference answer.

Question: {question}

Reference Answer: {reference}

Predicted Answer: {prediction}

Rate the predicted answer on a scale of 0 to 5 based on:
- Semantic similarity to the reference
- Factual correctness
- Completeness of the answer
- Relevance to the question

Scoring criteria:
5 - Perfect or nearly perfect match
4 - Very good, minor differences
3 - Acceptable, captures main points but missing some details
2 - Partially correct, significant gaps
1 - Mostly incorrect, some relevant information
0 - Completely wrong or irrelevant

Provide ONLY a single number (0-5) as your response, nothing else."""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that evaluates answers and provides numerical scores."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Extract score from response
    try:
        score = float(response.strip().split()[0])
        score = max(0, min(5, score))  # Clamp between 0 and 5
    except:
        print(f"Warning: Could not parse score from response: {response}. Using 2.5 as default.")
        score = 2.5
    
    return score

def calculate_metrics(predictions, references, questions=None, use_llm_judge=True):
    """
    Calculate BERTScore, ROUGE-L, METEOR, BLEU-4, and LLM-as-a-judge metrics
    """
    results = {
        'bertscore': {'precision': [], 'recall': [], 'f1': []},
        'rouge_l': [],
        'meteor': [],
        'bleu_4': [],
        'llm_judge': []
    }
    
    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    
    # Load LLM judge model if needed
    judge_model, judge_tokenizer = None, None
    if use_llm_judge and questions is not None:
        judge_model, judge_tokenizer = load_judge_model()
    
    print("Calculating metrics for each sample...")
    for idx, (pred, ref) in enumerate(tqdm(zip(predictions, references), total=len(predictions))):
        # Tokenize for BLEU and METEOR
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())
        
        # ROUGE-L
        rouge_scores = rouge.score(ref, pred)
        results['rouge_l'].append(rouge_scores['rougeL'].fmeasure)
        
        # METEOR
        meteor = meteor_score([ref_tokens], pred_tokens)
        results['meteor'].append(meteor)
        
        # BLEU-4
        bleu = sentence_bleu([ref_tokens], pred_tokens, 
                            smoothing_function=smoothie,
                            weights=(0.25, 0.25, 0.25, 0.25))
        results['bleu_4'].append(bleu)
        
        # LLM-as-a-judge
        if use_llm_judge and questions is not None and judge_model is not None:
            question = questions[idx]
            llm_score = llm_judge_score(question, pred, ref, judge_model, judge_tokenizer)
            results['llm_judge'].append(llm_score)
        
        # Clear cache periodically
        if (idx + 1) % 50 == 0 and judge_model is not None:
            torch.cuda.empty_cache()
    
    # BERTScore (calculate in batch for efficiency)
    print("Calculating BERTScore...")
    P, R, F1 = bert_score(predictions, references, lang='en', verbose=True)
    results['bertscore']['precision'] = P.tolist()
    results['bertscore']['recall'] = R.tolist()
    results['bertscore']['f1'] = F1.tolist()
    
    # Clean up judge model
    if judge_model is not None:
        del judge_model
        del judge_tokenizer
        torch.cuda.empty_cache()
    
    return results

def aggregate_metrics(results):
    """
    Calculate mean and std for all metrics
    """
    aggregated = {
        'bertscore': {
            'precision': {
                'mean': np.mean(results['bertscore']['precision']),
                'std': np.std(results['bertscore']['precision'])
            },
            'recall': {
                'mean': np.mean(results['bertscore']['recall']),
                'std': np.std(results['bertscore']['recall'])
            },
            'f1': {
                'mean': np.mean(results['bertscore']['f1']),
                'std': np.std(results['bertscore']['f1'])
            }
        },
        'rouge_l': {
            'mean': np.mean(results['rouge_l']),
            'std': np.std(results['rouge_l'])
        },
        'meteor': {
            'mean': np.mean(results['meteor']),
            'std': np.std(results['meteor'])
        },
        'bleu_4': {
            'mean': np.mean(results['bleu_4']),
            'std': np.std(results['bleu_4'])
        }
    }
    
    # Add LLM judge scores if available
    if results['llm_judge']:
        aggregated['llm_judge'] = {
            'mean': np.mean(results['llm_judge']),
            'std': np.std(results['llm_judge'])
        }
    
    return aggregated

def main():
    # Load test results with model answers
    # input_json_path = "/datastore/clc_hcmus/ZaAIC/CS412-CV-FinalProject/HAD/questions/test_with_answers.json"
    # output_json_path = "/datastore/clc_hcmus/ZaAIC/CS412-CV-FinalProject/HAD/questions/test_evaluation_results_with_llm_judge.json"
    input_json_path = "/datastore/clc_hcmus/ZaAIC/CS412-CV-FinalProject/HAD/questions/test_with_answers.json"
    output_json_path = "/datastore/clc_hcmus/ZaAIC/CS412-CV-FinalProject/HAD/questions/test_evaluation_results_with_llm_judge.json"
    
    
    print(f"Loading results from {input_json_path}")
    with open(input_json_path, "r") as f:
        test_data = json.load(f)
    
    print(f"Total samples: {len(test_data)}")
    
    # Extract predictions, references, and questions
    predictions = []
    references = []
    questions = []
    valid_samples = []
    
    for item in test_data:
        model_answer = item.get("model_answer", "")
        ground_truth = item.get("QA", {}).get("a", "")
        question = item.get("question", "")
        
        # Skip if error or empty
        if model_answer.startswith("ERROR") or not model_answer or not ground_truth:
            continue
        
        predictions.append(model_answer)
        references.append(ground_truth)
        questions.append(question)
        valid_samples.append(item)
    
    print(f"Valid samples for evaluation: {len(valid_samples)}")
    
    # Calculate metrics (including LLM-as-a-judge)
    results = calculate_metrics(predictions, references, questions, use_llm_judge=True)
    
    # Aggregate metrics
    aggregated = aggregate_metrics(results)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nBERTScore:")
    print(f"  Precision: {aggregated['bertscore']['precision']['mean']:.4f} ± {aggregated['bertscore']['precision']['std']:.4f}")
    print(f"  Recall:    {aggregated['bertscore']['recall']['mean']:.4f} ± {aggregated['bertscore']['recall']['std']:.4f}")
    print(f"  F1:        {aggregated['bertscore']['f1']['mean']:.4f} ± {aggregated['bertscore']['f1']['std']:.4f}")
    print(f"\nROUGE-L:   {aggregated['rouge_l']['mean']:.4f} ± {aggregated['rouge_l']['std']:.4f}")
    print(f"METEOR:    {aggregated['meteor']['mean']:.4f} ± {aggregated['meteor']['std']:.4f}")
    print(f"BLEU-4:    {aggregated['bleu_4']['mean']:.4f} ± {aggregated['bleu_4']['std']:.4f}")
    
    if 'llm_judge' in aggregated:
        print(f"LLM Judge: {aggregated['llm_judge']['mean']:.4f} ± {aggregated['llm_judge']['std']:.4f} (out of 5)")
    
    print("="*50)
    
    # Save detailed results
    output_data = {
        'aggregated_metrics': aggregated,
        'per_sample_metrics': results,
        'num_samples': len(valid_samples)
    }
    
    print(f"\nSaving evaluation results to {output_json_path}")
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()