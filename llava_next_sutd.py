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
import torchvision.transforms as T
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


def choose_answer(
    question: str,
    choices: list[str],
    keyframes: list,
    top_k_frames: list = None,
    relevant_law_sections: list = None,
    keyframe_descriptions: list = None,
    output_path: str = None,
    device: str = "cuda:7",
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
    tokenizer, model, image_processor = _load_model(device=device)

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
        .to(device)
        .bfloat16()
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
    full_question = (
        f"You are analyzing dashcam footage from a vehicle. "
        f"Carefully examine the sequence of images showing the traffic situation.\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choices_text}\n\n"
        f"Instructions:\n"
        f"- Analyze the traffic signs, road markings, vehicle positions, and traffic conditions in the images\n"
        f"- Consider traffic laws and road safety regulations\n"
        f"- Select the most appropriate answer based on the visual evidence\n"
        f"- Respond with ONLY the number ({letter_options}) of your chosen answer\n"
        f"- Do not provide explanations or additional text\n\n"
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
    mode="test",
):
    """Process all videos in dataset and generate answers.

    Args:
        video_dir (str): Directory containing videos
        questions_path (str): Path to questions JSON file
        output_dir (str): Directory to save output
        num_frames (int): Number of frames to extract per video
    """
    # Load questions
    questions_data = []

    with open(questions_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            # if line_num > 100:
            #     break
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
            answers.append("0")  # Default answer for missing video
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
                question=q_body, choices=choices, keyframes=keyframes, device=device
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
            answers.append("0")  # Default answer on error
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
        description="Process video dataset and generate answers using LLaVA-NeXT."
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
        help="Device to run the model on (default: cuda)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    _load_model(device=args.device)

    # Process dataset
    process_dataset(
        video_dir=args.video_dir,
        questions_path=args.questions_path,
        keyframe_dir=args.keyframe_dir,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        katna_extraction=args.katna_extraction,
        device=args.device,
        mode=args.mode,
    )


# # For test set
# python llava_next_sutd.py --video_dir ./SUTD/videos/ --questions_path ./SUTD/questions/R3_test.jsonl --output_dir ./SUTD/outputs_100/ --keyframe_dir ./SUTD/keyframes/ --device cuda:5 --num_frames 64 --mode test
# python llava_next_sutd.py --video_dir ./SUTD/videos/ --questions_path ./SUTD/questions/R3_test.jsonl --output_dir ./SUTD/outputs_6052/ --keyframe_dir ./SUTD/keyframes/ --device cuda:5 --num_frames 64 --mode test
# python llava_next_sutd.py --video_dir ./SUTD/videos/ --questions_path ./SUTD/questions/R3_test.jsonl --output_dir ./SUTD/outputs/ --keyframe_dir ./SUTD/keyframes_augment/ --device cuda:2 --num_frames 8 --mode test
