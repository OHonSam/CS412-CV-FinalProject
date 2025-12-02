from PIL import Image
import numpy as np
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from src.extract_keyframes import extract_keyframes
from loguru import logger
import pandas as pd
import av
import torch
import warnings
import copy
import gc
import os
import json
import traceback

warnings.filterwarnings("ignore")

_model = None
_tokenizer = None
_image_processor = None
_current_device = None

def _load_model(device: str = "cuda:7"):
    """Load the LLaVA-Video model (singleton pattern).

    Model is kept in memory across multiple video processing for efficiency.
    """
    global _model, _processor, _tokenizer, _image_processor, _current_device

    if _model is None or _current_device != device:
        if _model is not None:
            # Clean up previous model to free memory
            del _model
            gc.collect()
            torch.cuda.empty_cache()

        print("Loading LLaVA-Video model...")
        pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
        model_name = "llava_qwen"

        _tokenizer, _model, _image_processor, max_length = load_pretrained_model(
            pretrained,
            None,
            model_name,
            torch_dtype="bfloat16",
            device_map=device,
        )

        _model = _model.to(device)

        _model.eval()
        _current_device = device
        print("LLaVA-Video model loaded successfully")

    return _tokenizer, _model, _image_processor


def generate_answer(
    question: str,
    keyframes: list,
    device: str = "cuda:7",
) -> str:
    """Generate free-form answer for HAD dataset.

    Args:
        question (str): The question about the video
        keyframes (list): List of image paths (keyframes from the video)
        device (str): CUDA device to use

    Returns:
        str: Generated answer text
    """
    # Load model
    tokenizer, model, image_processor = _load_model(device=device)

    # Load images as PIL
    pil_images = []
    for keyframe_path in keyframes:
        try:
            if not os.path.exists(keyframe_path):
                print(f"Warning: Keyframe not found: {keyframe_path}")
                continue
            img = Image.open(keyframe_path).convert("RGB")
            pil_images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load image {keyframe_path}: {e}")
            continue

    if not pil_images:
        print("Warning: No valid keyframes loaded, returning error message")
        return "ERROR: No valid keyframes extracted from video"

    # ✅ Initialize variables for cleanup
    video = None
    input_ids = None
    output_ids = None

    try:
        # Preprocess frames    
        processed = image_processor.preprocess(pil_images, return_tensors="pt")
        video = processed["pixel_values"].to(device).bfloat16()
        video = [video]

        # Prepare prompt for HAD dataset (traffic context)
        # full_question = (
        #     f"You are analyzing dashcam footage from a vehicle. "
        #     f"Carefully examine the sequence of images showing the traffic situation.\n\n"
        #     f"Question: {question}\n\n"
        #     f"Instructions:\n"
        #     f"- Analyze the traffic signs, road markings, vehicle positions, and traffic conditions\n"
        #     f"- Consider traffic laws and road safety regulations\n"
        #     f"- Provide a clear and concise answer based on the visual evidence\n"
        #     f"- Be specific about what you observe in the images\n\n"
        #     f"Answer:"
        # )
        full_question = (
            f"You are an expert traffic analyst examining dashcam footage from a vehicle. "
            f"Analyze the sequence of frames carefully to understand the traffic scenario.\n\n"
            
            f"CONTEXT:\n"
            f"- This is real dashcam footage showing traffic situations\n"
            f"- Pay attention to temporal changes across frames\n"
            f"- Consider the driver's perspective and decision-making context\n\n"
            
            f"QUESTION: {question}\n\n"
            
            f"ANALYSIS GUIDELINES:\n"
            f"1. Visual Elements:\n"
            f"   - Identify all vehicles, pedestrians, and road users\n"
            f"   - Observe traffic signs, signals, and road markings\n"
            f"   - Note road conditions, weather, and lighting\n"
            f"   - Track movements and trajectories across frames\n\n"
            
            f"2. Traffic Rules & Safety:\n"
            f"   - Apply relevant traffic laws and regulations\n"
            f"   - Identify violations or risky behaviors\n"
            f"   - Consider right-of-way rules\n"
            f"   - Evaluate safe following distances and speeds\n\n"
            
            f"3. Situational Awareness:\n"
            f"   - Assess potential hazards or risks\n"
            f"   - Consider what the driver should do next\n"
            f"   - Evaluate whether actions are appropriate for the situation\n"
            f"   - Think about defensive driving principles\n\n"
            
            f"RESPONSE FORMAT:\n"
            f"- Provide a clear, concise answer (2-4 sentences)\n"
            f"- Base your answer strictly on visible evidence in the frames\n"
            f"- Be specific: mention what you see (e.g., 'the white sedan', 'the traffic light')\n"
            f"- Explain the reasoning behind your answer\n"
            f"- Use objective, factual language\n\n"
            
            f"Answer:"
        )
        
        # Build conversation
        conv_template = "qwen_1_5"
        question_with_token = DEFAULT_IMAGE_TOKEN + f"\n{full_question}"

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question_with_token)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        # Tokenize input
        input_ids = (
            tokenizer_image_token(
                prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(device)
        )

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=video,
                modalities=["video"],
                do_sample=True,
                temperature=0.1,
                max_new_tokens=256,
                top_p=0.1,
                num_beams=1,
            )

        # Decode response
        text_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA Out of Memory: {e}")
        print("Try reducing --num_frames or using a smaller batch size")
        text_output = "ERROR: CUDA Out of Memory"
        
    except Exception as e:
        print(f"Error during generation: {e}")
        traceback.print_exc()
        text_output = f"ERROR: Generation failed - {str(e)}"
    
    finally:
        if video is not None:
            del video
        if input_ids is not None:
            del input_ids
        if output_ids is not None:
            del output_ids
        
        gc.collect()
        torch.cuda.empty_cache()

    return text_output


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def extract_keyframes_from_video(
    video_path: str, num_frames: int = 8, temp_dir: str = "./temp_keyframes"
) -> list:
    """Extract keyframes from video and save temporarily.

    Args:
        video_path (str): Path to video file
        num_frames (int): Number of frames to extract
        temp_dir (str): Directory to save temporary keyframes

    Returns:
        list: List of keyframe paths
    """
    os.makedirs(temp_dir, exist_ok=True)

    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
    clip = read_video_pyav(container, indices)

    keyframes = []
    video_name = Path(video_path).stem

    for i, frame in enumerate(clip):
        img = Image.fromarray(frame)
        img_path = os.path.join(temp_dir, f"{video_name}_frame_{i}.png")
        img.save(img_path)
        keyframes.append(img_path)

    container.close()
    return keyframes


def process_had_dataset(
    video_dir: str,
    questions_path: str,
    keyframe_dir: str,
    output_dir: str,
    num_frames: int = 8,
    device: str = "cuda:7",
    katna_extraction: bool = False,
    mode: str = "test",
):
    """Process HAD dataset and generate answers."""
    # Load questions
    print(f"Loading questions from {questions_path}")
    with open(questions_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Total questions: {len(test_data)}")
    print(f"Device: {device}")
    print(f"Frames per video: {num_frames}")

    print("\nPre-loading model...")
    _load_model(device=device)
    
    initial_vram = torch.cuda.memory_allocated(device=device) / 1024**3
    print(f"Initial VRAM after model load: {initial_vram:.2f} GB")

    output_path = os.path.join(output_dir, f"{mode}_with_answers.json")
    
    # Create temp directory for keyframes
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_keyframes")
    os.makedirs(temp_dir, exist_ok=True)

    # Process each question
    results = []
    failed_count = 0
    
    for idx, item in enumerate(tqdm(test_data, desc="Processing videos")):
        try:
            # Get video info
            video_id = item.get("video_id")
            video_path = os.path.join(video_dir, video_id)
            
            # Get question
            qa_pair = item.get("QA", {})
            question = qa_pair.get("q", "")
            
            logger.debug(f"Processing {video_id}: {question}")
            
            # Check if video exists
            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")
                item["model_answer"] = "ERROR: Video not found"
                results.append(item)
                failed_count += 1
                continue
            
            # Extract keyframes
            try:
                keyframes = extract_keyframes_from_video(video_path, num_frames, temp_dir)
            except Exception as e:
                print(f"Error extracting keyframes from {video_id}: {e}")
                item["model_answer"] = f"ERROR: Keyframe extraction failed - {str(e)}"
                results.append(item)
                failed_count += 1
                continue
            
            # Generate answer
            answer = generate_answer(
                question=question,
                keyframes=keyframes,
                device=device
            )
            
            # Check if generation failed
            if answer.startswith("ERROR:"):
                failed_count += 1
            
            # Add model answer to the item
            item["model_answer"] = answer
            results.append(item)
            
            for keyframe in keyframes:
                if os.path.exists(keyframe):
                    os.remove(keyframe)
            
            if (idx + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
            if (idx + 1) % 10 == 0:
                vram_gb = torch.cuda.memory_allocated(device=device) / 1024**3
                peak_vram_gb = torch.cuda.max_memory_allocated(device=device) / 1024**3
                success_rate = ((idx + 1 - failed_count) / (idx + 1)) * 100
                print(f"\nProgress: {idx + 1}/{len(test_data)}")
                print(f"  Current VRAM: {vram_gb:.2f} GB")
                print(f"  Peak VRAM: {peak_vram_gb:.2f} GB")
                print(f"  Success rate: {success_rate:.1f}%")
            
        except Exception as e:
            print(f"\nError processing {video_id}: {str(e)}")
            traceback.print_exc()
            item["model_answer"] = f"ERROR: {str(e)}"
            results.append(item)
            failed_count += 1

    # Save results
    print(f"\nSaving results to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Clean up temp directory
    try:
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    except OSError:
        pass  # Directory not empty

    # Print final statistics
    final_vram = torch.cuda.memory_allocated(device=device) / 1024**3
    peak_vram = torch.cuda.max_memory_allocated(device=device) / 1024**3
    success_count = len(results) - failed_count
    
    print(f"\n{'='*60}")
    print(f"Completed! Results saved to {output_path}")
    print(f"{'='*60}")
    print(f"Total videos: {len(results)}")
    print(f"Successful: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"Failed: {failed_count} ({failed_count/len(results)*100:.1f}%)")
    print(f"Final VRAM: {final_vram:.2f} GB")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Process HAD video dataset and generate answers using LLaVA-Video."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Path to the input video directory (e.g., ./HAD/videos/test/)",
    )
    parser.add_argument(
        "--questions_path",
        type=str,
        required=True,
        help="Path to the questions JSON file (e.g., ./HAD/questions/test.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--keyframe_dir",
        type=str,
        default="./keyframes/",
        help="Path to the keyframe output directory",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to extract from each video (default: 8)",
    )
    parser.add_argument(
        "--katna_extraction",
        action="store_true",
        help="Use Katna for keyframe extraction",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        help="Dataset mode: train, val, or test (default: test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:7",
        help="Device to run the model on (default: cuda:7)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process dataset
    process_had_dataset(
        video_dir=args.video_dir,
        questions_path=args.questions_path,
        keyframe_dir=args.keyframe_dir,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        katna_extraction=args.katna_extraction,
        device=args.device,
        mode=args.mode,
    )


# Example usage:
# python llava_next_had_2.py --video_dir ./HAD/videos/test/ --questions_path ./HAD/questions/test.json --output_dir ./HAD/outputs/ --keyframe_dir ./HAD/keyframes/ --device cuda:4 --num_frames 8 --mode test