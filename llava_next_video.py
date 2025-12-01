# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import os
import sys
import warnings
from decord import VideoReader, cpu
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda:7"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

video_path = os.path.join("videos", "0ad45230_447_clip_006_0031_0037_N.mp4")

max_frames_num = 64
video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
video = [video]
conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."

final_question = "Trong video này, khi tiến đến gần vạch kẻ đường dành cho người đi bộ, người lái xe phải thực hiện hành động ưu tiên nào sau đây?"
choices = [
            "A. Tăng tốc độ để nhanh chóng đi qua trước khi có người sang đường.",
            "B. Giảm tốc độ, quan sát và sẵn sàng dừng lại nhường đường.",
            "C. Bấm còi liên tục để cảnh báo người đi bộ không được qua đường.",
            "D. Giữ nguyên tốc độ và chỉ dừng lại khi có người đã đi vào lòng đường."
            ]

letter_options = ", ".join([choice.split(".")[0] for choice in choices])
choices_text = "\n".join(choices)

full_question = (
    f"You are analyzing dashcam footage from a vehicle. "
    f"Carefully examine the sequence of images showing the traffic situation.\n\n"
    f"Question: {final_question}\n\n"
    f"Choices:\n{choices_text}\n\n"
    f"Instructions:\n"
    f"- Analyze the traffic signs, road markings, vehicle positions, and traffic conditions in the images\n"
    f"- Consider Vietnam traffic laws and road safety regulations\n"
    f"- Select the most appropriate answer based on the visual evidence\n"
    f"- Respond with ONLY the letter ({letter_options}) of your chosen answer\n"
    f"- Do not provide explanations or additional text\n\n"
    f"Answer:"
)

final_question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{full_question}"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], final_question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
cont = model.generate(
    input_ids,
    images=video,
    modalities= ["video"],
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print(text_outputs)
