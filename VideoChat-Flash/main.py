import torch
import os
import sys
import json
from tqdm import tqdm

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
    
    try:
        frames = load_video(video_path, num_frames=64)
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None
    
    pil_frames = [Image.fromarray(frame) for frame in frames]
    processed_frames = image_processor.preprocess(pil_frames, return_tensors='pt')['pixel_values']
    processed_frames = processed_frames.to(device, dtype=torch.float16)
    
    prompt_text = f"{question}\n\n"
    for choice in choices:
        prompt_text += f"{choice}\n"
    prompt_text += "\nAnswer with the letter only (A, B, C, or D)."
    
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt_text)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(
        full_prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[processed_frames],
            modalities=["video"],
            do_sample=False,
            max_new_tokens=32,
            use_cache=True,
        )
    
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
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
    
    for letter in ['A', 'B', 'C', 'D']:
        if answer_upper.startswith(letter):
            return letter
    
    for letter in ['A', 'B', 'C', 'D']:
        if letter in answer_upper:
            return letter
    
    return None


def run_benchmark(model, tokenizer, image_processor, video_dir, question_path, output_path=None, log_interval=100):
    """
    Run benchmark on all questions.
    """
    device = next(model.parameters()).device
    
    print(f"Loading questions from: {question_path}")
    questions = load_questions_jsonl(question_path)
    print(f"Loaded {len(questions)} questions\n")
    
    results = []
    correct = 0
    total = 0
    skipped = 0
    
    accuracy_by_type = {}
    
    for idx, item in enumerate(tqdm(questions, desc="Processing")):
        video_name = item['video']
        video_path = os.path.join(video_dir, video_name)
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            skipped += 1
            continue
        
        question = item['question']
        choices = item['choices']
        correct_answer = item['answer']
        q_type = item.get('q_type', 'unknown')
        
        model_answer = run_single_inference(
            model, tokenizer, image_processor,
            video_path, question, choices, device
        )
        
        predicted = extract_letter(model_answer)
        
        is_correct = predicted == correct_answer
        if is_correct:
            correct += 1
        total += 1
        
        if q_type not in accuracy_by_type:
            accuracy_by_type[q_type] = {'correct': 0, 'total': 0}
        accuracy_by_type[q_type]['total'] += 1
        if is_correct:
            accuracy_by_type[q_type]['correct'] += 1
        
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
        
        if (idx + 1) % log_interval == 0:
            current_accuracy = correct / total if total > 0 else 0
            print(f"\n{'='*60}")
            print(f"PROGRESS LOG - Question {idx + 1}/{len(questions)}")
            print(f"{'='*60}")
            print(f"Running accuracy: {correct}/{total} ({current_accuracy*100:.2f}%)")
            print(f"Skipped so far: {skipped}")
            
            print(f"\nAccuracy by type so far:")
            for qt, stats in sorted(accuracy_by_type.items()):
                type_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"  {qt}: {stats['correct']}/{stats['total']} ({type_acc*100:.2f}%)")
            
            print(f"\n--- Question #{idx + 1} Details ---")
            print(f"Video: {video_name}")
            print(f"Question: {question}")
            print(f"Choices: {choices}")
            print(f"Correct Answer: {correct_answer}")
            print(f"Model Answer: {model_answer}")
            print(f"Predicted: {predicted}")
            print(f"Is Correct: {'✓' if is_correct else '✗'}")
            print(f"{'='*60}\n")
            
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
    
    accuracy = correct / total if total > 0 else 0
    
    print("\n" + "="*60)
    print("FINAL BENCHMARK RESULTS")
    print("="*60)
    print(f"Total questions: {total}")
    print(f"Skipped (video not found): {skipped}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("="*60)
    
    print("\nFINAL ACCURACY BY QUESTION TYPE:")
    print("-"*40)
    for q_type, stats in sorted(accuracy_by_type.items()):
        type_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {q_type}: {stats['correct']}/{stats['total']} ({type_acc*100:.2f}%)")
    print("="*60)
    
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
    
    return results, accuracy


def main():

    model_path = "OpenGVLab/VideoChat-Flash-Qwen2-7B_res448"
    video_dir = "datasets/videos"
    question_path = "datasets/R2_test.jsonl"
    output_path = "./results.json"
    log_interval = 100  # Log every N questions

    
    if not os.path.exists(video_dir):
        print(f"ERROR: Video directory not found: {video_dir}")
        return
    
    if not os.path.exists(question_path):
        print(f"ERROR: Question file not found: {question_path}")
        return
    
    print("Loading model...")
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
    
    results, accuracy = run_benchmark(
        model, tokenizer, image_processor,
        video_dir, question_path, output_path,
        log_interval=log_interval
    )


if __name__ == "__main__":
    main()