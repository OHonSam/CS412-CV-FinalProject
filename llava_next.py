import av
import torch
import warnings
from PIL import Image
import numpy as np
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
import gc
import av
from pathlib import Path    

warnings.filterwarnings("ignore")

# Global model variables (loaded once)
_model = None
_processor = None
_tokenizer = None
_image_processor = None


def _load_model():
    """Load the LLaVA-Video model (singleton pattern).

    Model is kept in memory across multiple video processing for efficiency.
    """
    global _model, _processor, _tokenizer, _image_processor

    if _model is None:
        print("Loading LLaVA-Video model...")
        pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
        model_name = "llava_qwen"
        device_map = "auto"

        _tokenizer, _model, _image_processor, max_length = load_pretrained_model(
            pretrained,
            None,
            model_name,
            torch_dtype="bfloat16",
            device_map=device_map
        )
        _model.eval()
        print("LLaVA-Video model loaded successfully")

    return _tokenizer, _model, _image_processor


def choose_answer(
    question: str,
    choices: list[str],
    keyframes: list,
    top_k_frames: list = None,
    relevant_law_sections: list = None,
    keyframe_descriptions: list = None,
    output_path: str = None
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
    tokenizer, model, image_processor = _load_model()

    # Load images from keyframe paths
    images = []
    for keyframe_path in keyframes:
        try:
            img = Image.open(keyframe_path).convert("RGB")
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
    video = image_processor.preprocess(video_frames, return_tensors="pt")[
        "pixel_values"].cuda().bfloat16()
    video = [video]

    # Determine number of choices and format accordingly
    num_choices = len(choices)
    choice_letters = ['A', 'B', 'C', 'D'][:num_choices]

    # Format choices with letters
    choices_text = "\n".join(choices)

    # Build letter options for instruction
    if num_choices == 2:
        letter_options = "A or B"
    elif num_choices == 4:
        letter_options = "A, B, C, or D"
    else:
        letter_options = " or ".join(choice_letters)

    # Prepare the optimized prompt for dashcam traffic scenarios
    full_question = (
        f"You are analyzing dashcam footage from a vehicle. "
        f"Carefully examine the sequence of images showing the traffic situation.\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choices_text}\n\n"
        f"Instructions:\n"
        f"- Analyze the traffic signs, road markings, vehicle positions, and traffic conditions in the images\n"
        f"- Consider Vietnam traffic laws and road safety regulations\n"
        f"- Select the most appropriate answer based on the visual evidence\n"
        f"- Respond with ONLY the letter ({letter_options}) of your chosen answer\n"
        f"- Do not provide explanations or additional text\n\n"
        f"Answer:"
    )

    print(f"Full question prompt:\n{full_question}")
    print(f"Number of keyframes: {len(images)}")
    print(f"Number of choices: {num_choices}")

    # Build conversation
    conv_template = "qwen_1_5"
    question_with_token = DEFAULT_IMAGE_TOKEN + f"\n{full_question}"

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question_with_token)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    # Tokenize input
    input_ids = tokenizer_image_token(
        prompt_question,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).cuda()

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
    text_output = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True)[0].strip()
    print(f"Model response: {text_output}")

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
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
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

if __name__ == "__main__":
    # Example usage

    question = "Trong video này, khi tiến đến gần vạch kẻ đường dành cho người đi bộ, người lái xe phải thực hiện hành động ưu tiên nào sau đây?"
    choices = [
                "A. Tăng tốc độ để nhanh chóng đi qua trước khi có người sang đường.",
                "B. Giảm tốc độ, quan sát và sẵn sàng dừng lại nhường đường.",
                "C. Bấm còi liên tục để cảnh báo người đi bộ không được qua đường.",
                "D. Giữ nguyên tốc độ và chỉ dừng lại khi có người đã đi vào lòng đường."
                ]

    # Get parent directory of the script
    script_dir = Path(__file__).parent
    video_dir = script_dir.parent / "ZaloAIC"
    # Construct video path relative to script directory
    video_path = video_dir / "input" / "traffic_buddy_train+public_test" / "train/videos/0ad45230_447_clip_006_0031_0037_N.mp4"
    container = av.open(video_path)

    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = read_video_pyav(container, indices)

    keyframes = []
    for i, frame in enumerate(clip):
        img = Image.fromarray(frame)
        img_path = f"keyframe_{i}.png"
        img.save(img_path)
        keyframes.append(img_path)

    # Inference
    answer_index = choose_answer(
        question=question,
        choices=choices,
        keyframes=keyframes,
    )