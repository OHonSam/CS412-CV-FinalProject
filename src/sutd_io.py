import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any

# SUTD question files in this project are typically stored as .jsonl where each line is a Python-list-like string.
# Example mapping (from src/prepare_instruct_entry.py):
# [record_id, vid_id, vid_filename, perspective, q_body, opt0, opt1, opt2, opt3, answer_idx, ...]
#
# This loader is defensive: it supports either a python-list string (ast.literal_eval) or JSON.

def _parse_line(line: str) -> Any:
    line = line.strip()
    if not line:
        return None
    # Try JSON first (strict)
    try:
        return json.loads(line)
    except Exception:
        pass
    # Fall back to python-literal format
    return ast.literal_eval(line)

def load_sutd_jsonl(path: str) -> List[list]:
    p = Path(path)
    items: List[list] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            item = _parse_line(line)
            if item is None:
                continue
            # Optional header row
            if isinstance(item, list) and len(item) > 0 and item[0] == "record_id":
                continue
            items.append(item)
    return items

def to_mcq4(item: list) -> Tuple[str, str, List[str], int]:
    """Return (record_id, video_filename, [opt0..opt3], answer_idx)."""
    # Indices follow the mapping used by llava_next_sutd.py and prepare_instruct_entry.py
    record_id = str(item[0])
    video_filename = str(item[2])
    question = str(item[4])
    choices = [str(item[5]), str(item[6]), str(item[7]), str(item[8])]
    # Answer index position can vary across variants; the common one in SUTD R3 files is item[9] or item[10]
    answer_idx = None
    for idx in [9, 10, 11]:
        if idx < len(item) and isinstance(item[idx], (int, str)):
            try:
                v = int(item[idx])
                if 0 <= v <= 3:
                    answer_idx = v
                    break
            except Exception:
                pass
    if answer_idx is None:
        # If not present, caller may be running on unlabeled test; use -1.
        answer_idx = -1
    return record_id, video_filename, question, choices, answer_idx
