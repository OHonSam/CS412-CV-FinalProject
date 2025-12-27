from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from src.extract_keyframes import extract_keyframes
from loguru import logger
from transformers import BitsAndBytesConfig
import numpy as np
import pandas as pd
import av
import torch
import warnings
import copy
import gc
import os
import json
import traceback
import torchvision.transforms as T

warnings.filterwarnings("ignore")

_model = None
_tokenizer = None
_image_processor = None
_current_device = None

os.environ["HF_HOME"] = "/datastore/clc_hcmus/ZaAIC/hf_cache"

def get_dashcam_augmentation():
    return T.Compose(
        [
            T.ToTensor(),
            T.ColorJitter(
                brightness=0.3,    # Handle night/day variations
                contrast=0.3,      # Handle fog/rain/glare
                saturation=0.2,    # Handle washed-out colors
                hue=0.1            # Minor color shifts
            ),
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),  # Handle rain blur
            T.RandomAutocontrast(p=0.2),  # Handle low contrast scenes
            T.ToPILImage(),
        ]
    )

def _load_model(device: str = "cuda:7", quantization: str = "4bit"):
    """Load the LLaVA-Video model with quantization (singleton pattern).

    Model is kept in memory across multiple video processing for efficiency.
    
    Args:
        device (str): Device to load model on
        quantization (str): Quantization mode - "4bit", "8bit", or None for full precision
    """
    global _model, _processor, _tokenizer, _image_processor, _current_device

    if _model is None or _current_device != device:
        if _model is not None:
            # Clean up previous model to free memory
            del _model
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Loading LLaVA-Video model with {quantization} quantization...")
        pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
        model_name = "llava_qwen"

        # Configure quantization
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            torch_dtype = torch.bfloat16
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            torch_dtype = torch.bfloat16
        else:
            quantization_config = None
            torch_dtype = "bfloat16"

        # Load model with quantization config
        _tokenizer, _model, _image_processor, max_length = load_pretrained_model(
            pretrained,
            None,
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            quantization_config=quantization_config,
        )

        # DO NOT use .to() for quantized models - device_map handles placement
        # The model is already on the correct device via device_map parameter
        
        _model.eval()
        _current_device = device
        print(f"\nFinal VRAM usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    return _tokenizer, _model, _image_processor


def choose_answer(
    question: str,
    choices: list[str],
    keyframes: list,
    top_k_frames: list = None,
    relevant_law_sections: list = None,
    keyframe_descriptions: list = None,
    output_path: str = None,
    device: str = "cuda:7",
    quantization: str = "4bit"
) -> int:
    """Choose the best answer index based on provided information.

    Args:
        question (str): The question related to the video.
        choices (list[str]): List of answer choices.
        keyframes (list): List of image paths (keyframes from the video).
        top_k_frames (list): Top-k relevant frames from the video (unused).
        relevant_law_sections (list): Relevant law sections (unused).
        keyframe_descriptions (list): Descriptions of the keyframes (unused).
        output_path (str): Path to save any output files (unused).

    Returns:
        int: Index of the selected answer choice.
    """
    # Load model
    tokenizer, model, image_processor = _load_model(device=device, quantization=quantization)
    model_device = next(model.parameters()).device

    if quantization in ["4bit", "8bit"]:
        compute_dtype = torch.float16  
    else:
        compute_dtype = torch.bfloat16

    # Load images from keyframe paths
    images = []
    for keyframe_path in keyframes:
        try:
            img = Image.open(keyframe_path).convert("RGB")
            img = get_dashcam_augmentation()(img)
            images.append(np.array(img))
        except Exception as e:
            print(f"Warning: Failed to load image {keyframe_path}: {e}")
            continue

    if not images:
        print("Warning: No valid keyframes loaded, returning default answer 0")
        return 0

    # Convert list of numpy arrays to single numpy array
    video_frames = np.stack(images)

    # Preprocess frames
    video = (
        image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
        .to(model_device, dtype=compute_dtype)
    )
    video = [video]

    # Determine number of choices and format accordingly
    num_choices = len(choices)

    # Format choices with letters
    choices_text = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])

    # Build letter options for instruction
    letter_options = ", ".join([str(i) for i in range(num_choices)])
    choice_letters = [str(i) for i in range(num_choices)]

    # Prepare the optimized prompt for dashcam traffic scenarios
    full_question = f"""
    You are a traffic-scene reasoning assistant for SUTD-TrafficQA. The images are ordered in time (earliest → latest).
    Your job is to answer a MULTIPLE-CHOICE question by selecting the single best option.

    SUTD-TrafficQA contains 6 reasoning task types:
    1) Basic Understanding: identify objects/attributes/actions/counts and simple event relations.
    2) Attribution: explain the cause(s) of an event/accident (the underlying factors).
    3) Event Forecasting: predict what will happen next / the likely outcome.
    4) Reverse Reasoning: infer what happened BEFORE the shown segment (antecedent events).
    5) Counterfactual Inference: reason about a hypothetical "if" condition that did NOT occur.
    6) Introspection: choose preventive / safer actions that could avoid an accident/congestion.
    (Use the task type that best matches the question; do NOT output the task type.) 
    These task types match the benchmark design. 

    EVIDENCE RULES (important):
    - Use ONLY visual evidence from the image sequence + basic traffic rules/physics.
    - If a choice claims "there is no X" / "no accident" / "no barrier", verify presence/absence across ALL frames.
    - Prefer the option most consistent with the full temporal sequence (not a single frame).
    - If two options seem plausible, choose the one best supported by visible causality/timing.

    TASK-SPECIFIC CHECKLIST (do silently):
    A) Basic Understanding:
    - Identify road type/scene (lanes, markings, signs, intersection), agents (cars/peds/bikes), counts, positions.
    - Confirm any queried attribute by checking multiple frames.

    B) Attribution (WHY did it happen?):
    - Find the earliest mistake/trigger that plausibly causes the outcome (illegal lane change, tailgating, sudden brake, red-light, obstruction, etc.).
    - Choose the most direct cause supported by the sequence, not generic priors.

    C) Event Forecasting (WHAT will happen?):
    - Use trajectories: relative speed, closing distance, lane alignment, right-of-way, available space.
    - Decide if collision/near-miss/merge/brake is likely given motion trends.

    D) Reverse Reasoning (WHAT happened earlier?):
    - Infer prior events from current state (vehicle stopped position, damage, skid/avoidance, unusual lane position).
    - Choose the option that best explains how the scene got into the observed state.

    E) Counterfactual Inference (WHAT IF ...):
    - Treat the "if" condition as a change to the scene; mentally simulate the most likely outcome under traffic rules/physics.
    - Decide whether the hypothetical would remove the cause, add space/time, or still lead to the event.

    F) Introspection (HOW to prevent?):
    - Pick the safest feasible preventive action BEFORE the critical moment (slow down, keep lane, increase distance, yield, earlier braking, etc.).
    - Prefer actions that directly interrupt the causal chain seen in the video.

    Now answer:

    Question: {question}

    Choices:
    {choices_text}

    Output format:
    - Respond with ONLY the number/letter from ({letter_options}) corresponding to your chosen option.
    - No explanation, no extra text.

    Answer:
    """.strip()


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
        .to(model_device)
    )

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=128,
        )

    # Decode response
    text_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
        0
    ].strip()

    # Clean up intermediate tensors to free GPU memory
    del video, input_ids, output_ids
    gc.collect()
    torch.cuda.empty_cache()

    # Parse the answer - prioritize exact letter matches at the start
    text_output_upper = text_output.upper()

    # First, check if response starts with a valid letter
    if text_output_upper and text_output_upper[0] in choice_letters:
        return choice_letters.index(text_output_upper[0])

    # Check first 10 characters for valid letters
    for i, choice_letter in enumerate(choice_letters):
        if choice_letter in text_output_upper[:10]:
            return i

    # Try to match with choice text
    for i, choice in enumerate(choices):
        if choice.lower() in text_output.lower():
            return i

    # Default to first choice if parsing fails
    print("Warning: Could not parse answer from model output, returning default 0")
    return 0


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


def process_dataset(
    video_dir: str,
    questions_path: str,
    keyframe_dir: str,
    output_dir: str,
    num_frames: int = 8,
    katna_extraction: bool = False,
    device: str = "cuda:7",
    quantization: str = "4bit",
    mode="test",
):
    """Process all videos in dataset and generate answers.

    Args:
        video_dir (str): Directory containing videos
        questions_path (str): Path to questions JSON file
        keyframe_dir (str): Directory for keyframes
        output_dir (str): Directory to save output
        num_frames (int): Number of frames to extract per video
        katna_extraction (bool): Use Katna for extraction
        device (str): Device to run on
        quantization (str): Quantization mode - "4bit", "8bit", or None
        mode (str): Dataset mode
    """
    # Load questions
    questions_data = []

    with open(questions_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line.strip():
                try:
                    data = json.loads(line)
                    if (
                        line_num == 0
                        and isinstance(data, list)
                        and data[0] == "record_id"
                    ):
                        print(f"Skipping header: {data}")
                        continue
                    questions_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue

    # Prepare output
    file_names = []
    answers = []
    record_ids = []
    validity_flags = []

    if katna_extraction:
        keyframes_dir = os.path.join(keyframe_dir, mode)
        os.makedirs(keyframes_dir, exist_ok=True)
    else:
        temp_dir = os.path.join(output_dir, "temp_keyframes")
        os.makedirs(temp_dir, exist_ok=True)

    # Process each video
    print(f"Processing {len(questions_data)} videos...")

    for item in tqdm(questions_data, desc=f"Processing {mode} dataset"):
        record_id = item[0]
        vid_filename = item[2]
        q_body = item[4]
        choices = [item[5], item[6], item[7], item[8]]
        logger.debug(f"Processing video: {vid_filename} with record_id: {record_id}")

        # Construct video path
        video_path = os.path.join(video_dir, vid_filename)

        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            record_ids.append(record_id)
            file_names.append(vid_filename)
            answers.append("0")
            validity_flags.append(False)
            continue

        try:
            # Extract keyframes
            if katna_extraction:
                keyframes = extract_keyframes(
                    video_path=video_path,
                    output_dir=keyframes_dir,
                    num_keyframes=num_frames,
                )
            else:
                keyframes = extract_keyframes_from_video(
                    video_path, 
                    num_frames, 
                    temp_dir
                )

            # Get answer
            answer_index = choose_answer(
                question=q_body, 
                choices=choices, 
                keyframes=keyframes, 
                device=device,
                quantization=quantization
            )

            answer_letter = str(answer_index)

            # Add to results
            record_ids.append(record_id)
            file_names.append(vid_filename)
            answers.append(answer_letter)
            validity_flags.append(True)

            # Clean up temporary keyframes
            if not katna_extraction:
                for keyframe in keyframes:
                    if os.path.exists(keyframe):
                        os.remove(keyframe)

        except Exception as e:
            print(f"Error processing {vid_filename}: {e}")
            traceback.print_exc()
            record_ids.append(record_id)
            file_names.append(vid_filename)
            answers.append("0")
            validity_flags.append(False)

    # Save results
    output_file = os.path.join(output_dir, f"{mode}_answer.csv")
    df = pd.DataFrame(
        {
            "id": record_ids,
            "filename": file_names,
            "answer": answers,
            "validity": validity_flags,
        }
    )
    df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")
    print(f"Processed {len(file_names)} videos")

    # Clean up temp directory
    if not katna_extraction:
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Process video dataset and generate answers using LLaVA-Video with quantization."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Path to the input video directory",
    )
    parser.add_argument(
        "--questions_path",
        type=str,
        required=True,
        help="Path to the questions JSON file",
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
        default="cuda:0",
        help="Device to run the model on (default: cuda:0)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="Quantization mode: 4bit, 8bit, or none (default: 4bit)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set quantization to None if "none" is specified
    quant_mode = None if args.quantization == "none" else args.quantization

    # Pre-load model
    _load_model(device=args.device, quantization=quant_mode)

    # Process dataset
    process_dataset(
        video_dir=args.video_dir,
        questions_path=args.questions_path,
        keyframe_dir=args.keyframe_dir,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        katna_extraction=args.katna_extraction,
        device=args.device,
        quantization=quant_mode,
        mode=args.mode,
    )

# Example usage with 4-bit quantization:
# python llava_next_sutd_quantized.py --video_dir ./SUTD/videos/ --questions_path ./SUTD/questions/R3_test.jsonl --output_dir ./SUTD/outputs_4bit/ --keyframe_dir ./SUTD/keyframes/ --device cuda:0 --num_frames 8 --mode test --quantization 4bit

# Example usage with 8-bit quantization:
# python llava_next_sutd_quantized.py --video_dir ./SUTD/videos/ --questions_path ./SUTD/questions/R3_test.jsonl --output_dir ./SUTD/outputs_8bit/ --keyframe_dir ./SUTD/keyframes/ --device cuda:0 --num_frames 8 --mode test --quantization 8bit
# python llava_next_sutd_quantized.py --video_dir ./SUTD/videos/ --questions_path ./SUTD/questions/R3_test.jsonl --output_dir ./SUTD/outputs_8bit_new_prompt/ --keyframe_dir ./SUTD/keyframes/ --device cuda:0 --num_frames 8 --mode test --quantization 8bit