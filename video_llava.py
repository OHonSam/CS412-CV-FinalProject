import requests
import numpy as np
import av
import torch
from huggingface_hub import hf_hub_download
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from PIL import Image

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

model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model = model.to(device)

prompt = "USER: <video>Why is this video funny? ASSISTANT:"
video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
container = av.open(video_path)

# sample uniformly 8 frames from the video
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
clip = read_video_pyav(container, indices)

inputs = processor(text=prompt, videos=clip, return_tensors="pt")
# inputs = inputs.to(device)
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=2075)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
# >>> 'USER:  Why is this video funny? ASSISTANT: The video is funny because the baby is sitting on the bed and reading a book, which is an unusual and amusing sight.Ъ'

# # Generate from images and videos mix
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# prompt = [
#     "USER: <image> How many cats are there in the image? ASSISTANT:",
#     "USER: <video>Why is this video funny? ASSISTANT:"
# ]
# inputs = processor(text=prompt, images=image, videos=clip, padding=True, return_tensors="pt")

# # Generate
# generate_ids = model.generate(**inputs, max_length=50)
# print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
# # >>> ['USER:   How many cats are there in the image? ASSISTANT: There are two cats in the image.\nHow many cats are sleeping on the couch?\nThere are', 'USER:  Why is this video funny? ASSISTANT: The video is funny because the baby is sitting on the bed and reading a book, which is an unusual and amusing']
