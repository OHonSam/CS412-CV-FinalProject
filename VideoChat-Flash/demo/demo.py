import torch
import os
import sys
import json
import random

# Add the parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llava-train_videochat'))

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
    """Load questions from JSONL file."""
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
                answer_letter = answer_map.get(answer, answer)
                
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


def demo():
    """Run a demo with a random question."""
    
    # Paths (relative to project root)
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    model_path = "OpenGVLab/VideoChat-Flash-Qwen2-7B_res448"
    video_dir = os.path.join(base_dir, "datasets", "videos")
    question_path = os.path.join(base_dir, "datasets", "R2_test.jsonl")
    
    # Check paths
    if not os.path.exists(video_dir):
        print(f"ERROR: Video directory not found: {video_dir}")
        return
    
    if not os.path.exists(question_path):
        print(f"ERROR: Question file not found: {question_path}")
        return
    
    # Load questions
    print("Loading questions...")
    questions = load_questions_jsonl(question_path)
    print(f"Loaded {len(questions)} questions\n")
    
    # Filter questions to only those with existing videos
    valid_questions = []
    for q in questions:
        video_path = os.path.join(video_dir, q['video'])
        if os.path.exists(video_path):
            valid_questions.append(q)
    
    if not valid_questions:
        print("ERROR: No valid questions found (videos missing)")
        return
    
    print(f"Found {len(valid_questions)} questions with available videos\n")
    
    # Load model
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
    device = next(model.parameters()).device
    print("Model loaded!\n")
    
    # Random select a question
    item = random.choice(valid_questions)
    video_path = os.path.join(video_dir, item['video'])
    
    print("=" * 60)
    print("RANDOM QUESTION DEMO")
    print("=" * 60)
    print(f"Video: {item['video']}")
    print(f"Question Type: {item['q_type']}")
    print(f"Perspective: {item['perspective']}")
    print(f"\nQuestion: {item['question']}")
    print("\nChoices:")
    for choice in item['choices']:
        print(f"  {choice}")
    print(f"\nCorrect Answer: {item['answer']}")
    print("-" * 60)
    
    # Run inference
    print("\nRunning model inference...")
    model_answer = run_single_inference(
        model, tokenizer, image_processor,
        video_path, item['question'], item['choices'], device
    )
    
    predicted = extract_letter(model_answer)
    is_correct = predicted == item['answer']
    
    print(f"\nModel Raw Output: {model_answer}")
    print(f"Predicted Answer: {predicted}")
    print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    print("=" * 60)


if __name__ == "__main__":
    demo()