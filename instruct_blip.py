from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import numpy as np
import av
from PIL import Image
import requests
from pathlib import Path

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

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

device = "cuda:2" if torch.cuda.is_available() else "cpu"
model.to(device)

script_dir = Path(__file__).parent
video_dir = script_dir.parent / "ZaloAIC"
# Construct video path relative to script directory
video_path = video_dir / "input" / "traffic_buddy_train+public_test" / "train/videos/0ad45230_447_clip_006_0031_0037_N.mp4"
container = av.open(video_path)

total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 2).astype(int)
clip = read_video_pyav(container, indices)

keyframes = []
for i, frame in enumerate(clip):
    img = Image.fromarray(frame)
    img_path = f"keyframe_{i}.png"
    img.save(img_path)
    keyframes.append(img_path)

question = "Trong video này, khi tiến đến gần vạch kẻ đường dành cho người đi bộ, người lái xe phải thực hiện hành động ưu tiên nào sau đây?"
choices = [
            "A. Giảm tốc độ, quan sát và sẵn sàng dừng lại nhường đường.",
            "B. Tăng tốc độ để nhanh chóng đi qua trước khi có người sang đường.",
            "C. Bấm còi liên tục để cảnh báo người đi bộ không được qua đường.",
            "D. Giữ nguyên tốc độ và chỉ dừng lại khi có người đã đi vào lòng đường."
            ]
choices_text = "\n".join(choices)
letter_options = ", ".join([choice.split(".")[0] for choice in choices])

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

inputs = processor(
    images=[Image.open(img_path) for img_path in keyframes], 
    text=[full_question] * len(keyframes), 
    return_tensors="pt"
    ).to(device)

outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=512,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)
