import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

# torchvision is convenient but sometimes fails to import in certain environments (binary op mismatch).
# We support a fallback minimal transform pipeline so the baseline still runs.
try:
    import torchvision.transforms as TVT
    _HAS_TORCHVISION = True
except Exception:
    TVT = None
    _HAS_TORCHVISION = False

try:
    import decord
    decord.bridge.set_bridge("torch")
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False

import av  # fallback video reader
import numpy as np

from .sutd_io import load_sutd_jsonl, to_mcq4
from .text_tokenizer import simple_tokenize, Vocab


@dataclass
class SUTDPaths:
    root: str
    videos_dir: str
    questions_path: str


def _uniform_indices(num_frames_total: int, num_frames: int) -> List[int]:
    if num_frames_total <= 0:
        return [0] * num_frames
    if num_frames_total <= num_frames:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (num_frames - num_frames_total)
    step = (num_frames_total - 1) / (num_frames - 1)
    return [int(round(i * step)) for i in range(num_frames)]


def _read_frames_decord(video_path: str, num_frames: int) -> torch.Tensor:
    vr = decord.VideoReader(video_path)
    idxs = _uniform_indices(len(vr), num_frames)
    frames = vr.get_batch(idxs)  # (T,H,W,3) torch uint8
    frames = frames.permute(0, 3, 1, 2).contiguous()  # (T,3,H,W)
    return frames


def _read_frames_pyav(video_path: str, num_frames: int) -> torch.Tensor:
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames

    if total is None or total <= 0:
        decoded = [f.to_image() for f in container.decode(video=0)]
        total = len(decoded)
        idxs = _uniform_indices(total, num_frames)
        selected = [decoded[i] for i in idxs]
    else:
        idxs = set(_uniform_indices(total, num_frames))
        selected = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in idxs:
                selected.append(frame.to_image())
        if len(selected) < num_frames and len(selected) > 0:
            selected += [selected[-1]] * (num_frames - len(selected))

    out = []
    for img in selected:
        arr = np.array(img.convert("RGB"))  # (H,W,3)
        out.append(torch.from_numpy(arr).permute(2, 0, 1))  # (3,H,W) uint8
    return torch.stack(out, dim=0)


def read_video_frames(video_path: str, num_frames: int) -> torch.Tensor:
    if _HAS_DECORD:
        return _read_frames_decord(video_path, num_frames)
    return _read_frames_pyav(video_path, num_frames)


def _resize_normalize_fallback(frames_u8: torch.Tensor, image_size: int) -> torch.Tensor:
    """frames_u8: (T,3,H,W) uint8 -> (T,3,image_size,image_size) float normalized."""
    x = frames_u8.float() / 255.0
    x = torch.nn.functional.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _build_torchvision_transform(image_size: int, train: bool, use_train_aug: bool):
    base = [TVT.Resize((image_size, image_size), antialias=True)]
    if train and use_train_aug:
        # conservative aug: no hue, no flip (traffic lights + left/right semantics)
        base += [
            TVT.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            TVT.RandomApply([TVT.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
        ]
    base += [
        TVT.ConvertImageDtype(torch.float32),
        TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return TVT.Compose(base)


class SUTDMCQ4Dataset(Dataset):
    """SUTD TrafficQA 4-choice setting dataset for CNN+LSTM baseline.

    Expects:
        SUTD/
          videos/
          questions/
            R3_train.jsonl / R3_val.jsonl / R3_test.jsonl
    """

    def __init__(
        self,
        questions_path: str,
        videos_dir: str,
        vocab: Vocab,
        max_q_len: int = 32,
        max_a_len: int = 32,
        num_frames: int = 16,
        image_size: int = 224,
        train: bool = True,
        use_train_aug: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.items = load_sutd_jsonl(questions_path)
        if max_samples is not None:
            self.items = self.items[:max_samples]
        self.videos_dir = videos_dir
        self.vocab = vocab
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.num_frames = num_frames
        self.image_size = image_size

        self.use_tv = _HAS_TORCHVISION
        self.transform = _build_torchvision_transform(image_size, train, use_train_aug) if self.use_tv else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record_id, vid_filename, question, choices, answer_idx = to_mcq4(self.items[idx])

        video_path = os.path.join(self.videos_dir, vid_filename)
        if not os.path.exists(video_path):
            frames = torch.zeros(self.num_frames, 3, self.image_size, self.image_size, dtype=torch.float32)
            valid = 0
        else:
            raw = read_video_frames(video_path, self.num_frames)  # uint8 (T,3,H,W)
            if self.use_tv:
                frames = torch.stack([self.transform(f) for f in raw], dim=0)
            else:
                frames = _resize_normalize_fallback(raw, self.image_size)
            valid = 1

        q_ids = self.vocab.encode(simple_tokenize(question), self.max_q_len)
        a_ids = [self.vocab.encode(simple_tokenize(c), self.max_a_len) for c in choices]

        return {
            "record_id": record_id,
            "frames": frames,  # (T,3,H,W)
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "a_ids": torch.tensor(a_ids, dtype=torch.long),  # (4,La)
            "label": torch.tensor(answer_idx, dtype=torch.long),
            "valid": torch.tensor(valid, dtype=torch.long),
            "vid_filename": vid_filename,
        }
