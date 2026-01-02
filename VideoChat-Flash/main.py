import torch
import os
import sys

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
    print(f"Loading video: {video_path}")
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    # Sample frames uniformly
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    
    print(f"Loaded {len(frames)} frames from {total_frames} total")
    return frames


def run_inference(model, tokenizer, image_processor, video_path, question, choices):
    """Run inference on a single video question."""
    
    device = next(model.parameters()).device
    
    # 1. Load video
    frames = load_video(video_path, num_frames=64)
    
    # 2. Process frames
    pil_frames = [Image.fromarray(frame) for frame in frames]
    processed_frames = image_processor.preprocess(pil_frames, return_tensors='pt')['pixel_values']
    processed_frames = processed_frames.to(device, dtype=torch.float16)
    print(f"Processed frames shape: {processed_frames.shape}")
    
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
    
    print("\n" + "="*50)
    print("PROMPT:")
    print(full_prompt)
    print("="*50 + "\n")
    
    # 5. Tokenize
    input_ids = tokenizer_image_token(
        full_prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).to(device)
    
    # 6. Generate
    print("Generating answer...")
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


def main():
    # ============ CONFIGURATION ============
    # Change these to your paths
    model_path = "OpenGVLab/VideoChat-Flash-Qwen2-7B_res448"
    video_path = "b_1a4411B7sb_clip_005.mp4"  # Your video file
    
    # Your multiple choice question
    question = "Which factors might have contributed to the accident?"
    choices = [
        "Traffic congestion",
        "Bad road surfaces",
        "Others",
        "Fatigue driving"
      ],
    correct_answer = "C"  # The correct answer for comparison
    # =======================================
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found at {video_path}")
        print("Please update the video_path variable with a valid path.")
        return
    
    # Load model
    print("Loading model (this may take a minute)...")
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
    
    # Run inference
    answer = run_inference(
        model, tokenizer, image_processor,
        video_path, question, choices
    )
    
    # Print results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Question: {question}")
    print(f"Model Answer: {answer}")
    print(f"Correct Answer: {correct_answer}")
    
    # Check if correct
    predicted_letter = None
    for letter in ['A', 'B', 'C', 'D']:
        if letter in answer.upper():
            predicted_letter = letter
            break
    
    if predicted_letter == correct_answer:
        print("✓ CORRECT!")
    else:
        print(f"✗ INCORRECT (predicted {predicted_letter})")


if __name__ == "__main__":
    main()