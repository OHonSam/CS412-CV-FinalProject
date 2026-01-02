import torch
import os
import sys
import json
from tqdm import tqdm

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'llava-train_videochat'))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from decord import VideoReader, cpu
import numpy as np
from PIL import Image


def load_video(video_path, num_frames=64):
    """Load video frames."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    # Sample frames uniformly
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    
    return frames


def load_questions_jsonl(question_path):
    """
    Load questions from JSONL file.
    
    Each line format:
    ["record_id", "vid_id", "vid_filename", "perspective", "q_body", "q_type", "option0", "option1", "option2", "option3", "answer"]
    """
    questions = []
    
    with open(question_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Parse the array format
                # ["record_id", "vid_id", "vid_filename", "perspective", "q_body", "q_type", "option0", "option1", "option2", "option3", "answer"]
                #      0           1           2              3            4         5          6         7          8          9          10
                
                record_id = data[0]
                vid_id = data[1]
                vid_filename = data[2]
                perspective = data[3]
                q_body = data[4]
                q_type = data[5]
                option0 = data[6]
                option1 = data[7]
                option2 = data[8]
                option3 = data[9]
                answer = data[10]
                
                # Convert answer index to letter (0 -> A, 1 -> B, etc.)
                answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', '0': 'A', '1': 'B', '2': 'C', '3': 'D'}
                answer_letter = answer_map.get(answer, answer)  # Keep as-is if already a letter
                
                question = {
                    'record_id': record_id,
                    'vid_id': vid_id,
                    'video': vid_filename,
                    'perspective': perspective,
                    'question': q_body,
                    'q_type': q_type,
                    'choices': [
                        f"A. {option0}",
                        f"B. {option1}",
                        f"C. {option2}",
                        f"D. {option3}"
                    ],
                    'answer': answer_letter
                }
                questions.append(question)
                
            except (json.JSONDecodeError, IndexError) as e:
                print(f"Warning: Failed to parse line: {line[:100]}... Error: {e}")
                continue
    
    return questions


def run_single_inference(model, tokenizer, image_processor, video_path, question, choices, device):
    """Run inference on a single video question."""
    
    # 1. Load video
    try:
        frames = load_video(video_path, num_frames=64)
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None
    
    # 2. Process frames
    pil_frames = [Image.fromarray(frame) for frame in frames]
    processed_frames = image_processor.preprocess(pil_frames, return_tensors='pt')['pixel_values']
    processed_frames = processed_frames.to(device, dtype=torch.float16)
    
    # 3. Build prompt
    prompt_text = f"{question}\n\n"
    for choice in choices:
        prompt_text += f"{choice}\n"
    prompt_text += "\nAnswer with the letter only (A, B, C, or D)."
    
    # 4. Use conversation template
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt_text)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    
    # 5. Tokenize
    input_ids = tokenizer_image_token(
        full_prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).to(device)
    
    # 6. Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[processed_frames],
            modalities=["video"],
            do_sample=False,
            max_new_tokens=32,
            use_cache=True,
        )
    
    # 7. Decode
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract answer after assistant marker
    if conv.roles[1] in full_output:
        answer = full_output.split(conv.roles[1])[-1].strip()
    else:
        answer = full_output.strip()
    
    return answer


def extract_letter(answer):
    """Extract the answer letter (A, B, C, D) from model output."""
    if answer is None:
        return None
    
    answer_upper = answer.upper()
    
    # Try to find a letter at the start
    for letter in ['A', 'B', 'C', 'D']:
        if answer_upper.startswith(letter):
            return letter
    
    # Try to find any letter
    for letter in ['A', 'B', 'C', 'D']:
        if letter in answer_upper:
            return letter
    
    return None


def run_benchmark(model, tokenizer, image_processor, video_dir, question_path, output_path=None, log_interval=100):
    """
    Run benchmark on all questions.
    """
    device = next(model.parameters()).device
    
    # Load questions from JSONL
    print(f"Loading questions from: {question_path}")
    questions = load_questions_jsonl(question_path)
    print(f"Loaded {len(questions)} questions\n")
    
    # Run inference on each question
    results = []
    correct = 0
    total = 0
    skipped = 0
    
    # Track accuracy by question type
    accuracy_by_type = {}
    
    for idx, item in enumerate(tqdm(questions, desc="Processing")):
        # Get video path
        video_name = item['video']
        video_path = os.path.join(video_dir, video_name)
        
        # Skip if video doesn't exist
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            skipped += 1
            continue
        
        # Get question and choices
        question = item['question']
        choices = item['choices']
        correct_answer = item['answer']
        q_type = item.get('q_type', 'unknown')
        
        # Run inference
        model_answer = run_single_inference(
            model, tokenizer, image_processor,
            video_path, question, choices, device
        )
        
        # Extract predicted letter
        predicted = extract_letter(model_answer)
        
        # Check if correct
        is_correct = predicted == correct_answer
        if is_correct:
            correct += 1
        total += 1
        
        # Track by question type
        if q_type not in accuracy_by_type:
            accuracy_by_type[q_type] = {'correct': 0, 'total': 0}
        accuracy_by_type[q_type]['total'] += 1
        if is_correct:
            accuracy_by_type[q_type]['correct'] += 1
        
        # Store result
        result = {
            'record_id': item.get('record_id', ''),
            'vid_id': item.get('vid_id', ''),
            'video': video_name,
            'perspective': item.get('perspective', ''),
            'question': question,
            'q_type': q_type,
            'choices': choices,
            'correct_answer': correct_answer,
            'model_answer': model_answer,
            'predicted': predicted,
            'is_correct': is_correct
        }
        results.append(result)
        
        # Log every log_interval questions
        if (idx + 1) % log_interval == 0:
            current_accuracy = correct / total if total > 0 else 0
            print(f"\n{'='*60}")
            print(f"PROGRESS LOG - Question {idx + 1}/{len(questions)}")
            print(f"{'='*60}")
            print(f"Running accuracy: {correct}/{total} ({current_accuracy*100:.2f}%)")
            print(f"Skipped so far: {skipped}")
            
            # Log accuracy by type so far
            print(f"\nAccuracy by type so far:")
            for qt, stats in sorted(accuracy_by_type.items()):
                type_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"  {qt}: {stats['correct']}/{stats['total']} ({type_acc*100:.2f}%)")
            
            # Log the current (100th) question details
            print(f"\n--- Question #{idx + 1} Details ---")
            print(f"Video: {video_name}")
            print(f"Question: {question}")
            print(f"Choices: {choices}")
            print(f"Correct Answer: {correct_answer}")
            print(f"Model Answer: {model_answer}")
            print(f"Predicted: {predicted}")
            print(f"Is Correct: {'✓' if is_correct else '✗'}")
            print(f"{'='*60}\n")
            
            # Save intermediate results
            if output_path:
                intermediate_output = output_path.replace('.json', f'_checkpoint_{idx + 1}.json')
                intermediate_data = {
                    'checkpoint': idx + 1,
                    'accuracy': current_accuracy,
                    'correct': correct,
                    'total': total,
                    'skipped': skipped,
                    'accuracy_by_type': {
                        k: {
                            'correct': v['correct'],
                            'total': v['total'],
                            'accuracy': v['correct'] / v['total'] if v['total'] > 0 else 0
                        }
                        for k, v in accuracy_by_type.items()
                    },
                    'results': results
                }
                with open(intermediate_output, 'w', encoding='utf-8') as f:
                    json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
                print(f"Checkpoint saved to: {intermediate_output}")
    
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL BENCHMARK RESULTS")
    print("="*60)
    print(f"Total questions: {total}")
    print(f"Skipped (video not found): {skipped}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("="*60)
    
    # Print accuracy by question type
    print("\nFINAL ACCURACY BY QUESTION TYPE:")
    print("-"*40)
    for q_type, stats in sorted(accuracy_by_type.items()):
        type_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {q_type}: {stats['correct']}/{stats['total']} ({type_acc*100:.2f}%)")
    print("="*60)
    
    # Save final results if output path provided
    if output_path:
        output_data = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'skipped': skipped,
            'accuracy_by_type': {
                k: {
                    'correct': v['correct'],
                    'total': v['total'],
                    'accuracy': v['correct'] / v['total'] if v['total'] > 0 else 0
                }
                for k, v in accuracy_by_type.items()
            },
            'results': results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nFinal results saved to: {output_path}")
    
    # Print some examples (first 5 incorrect ones)
    print("\n" + "="*60)
    print("SAMPLE INCORRECT PREDICTIONS:")
    print("="*60)
    incorrect = [r for r in results if not r['is_correct']][:5]
    for r in incorrect:
        print(f"\nRecord ID: {r['record_id']}")
        print(f"Video: {r['video']}")
        print(f"Question Type: {r['q_type']}")
        print(f"Question: {r['question']}")
        print(f"Choices: {r['choices']}")
        print(f"Correct: {r['correct_answer']}, Predicted: {r['predicted']}")
        print(f"Model output: {r['model_answer']}")
    
    return results, accuracy


def main():
    # ============ CONFIGURATION ============
    model_path = "OpenGVLab/VideoChat-Flash-Qwen2-7B_res448"
    video_dir = "/kaggle/input/sutd-traffic-video-qa/SUTD/videos"  # Directory containing videos
    question_path = "/root/CS412-CV-FinalProject/R2_test.jsonl"  # JSONL file with questions
    output_path = "/root/CS412-CV-FinalProject/results.json"  # Output file for results
    log_interval = 100  # Log every N questions
    # =======================================
    
    # Check paths exist
    if not os.path.exists(video_dir):
        print(f"ERROR: Video directory not found: {video_dir}")
        return
    
    if not os.path.exists(question_path):
        print(f"ERROR: Question file not found: {question_path}")
        return
    
    # Load model
    print("="*60)
    print("Loading model...")
    print("="*60)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="videochat-flash",
        device_map="auto",
        multimodal=True,
        torch_dtype=torch.float16
    )
    model.eval()
    print("Model loaded!\n")
    
    # Run benchmark
    results, accuracy = run_benchmark(
        model, tokenizer, image_processor,
        video_dir, question_path, output_path,
        log_interval=log_interval
    )


if __name__ == "__main__":
    main()