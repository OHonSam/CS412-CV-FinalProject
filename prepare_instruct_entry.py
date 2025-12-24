import json
import os
import ast
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(SCRIPT_DIR).parent
INPUT_FILE = PROJECT_ROOT / "SUTD" / "questions" / "R3_train.jsonl"       
OUTPUT_FILE = PROJECT_ROOT / "sutd_llava_train.json"
VIDEO_FOLDER = PROJECT_ROOT / "SUTD" / "videos"

def format_prompt(question, choices):
    """
    Formats the input exactly like your inference script (llava_next_sutd.py).
    This ensures the model sees the same pattern during training and testing.
    """
    # Create "0. OptionA", "1. OptionB", etc.
    # We preserve empty options so the index (0,1,2,3) remains correct.
    choices_text = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
    
    # allowed_indices = [str(i) for i in range(len(choices))]
    letter_options = ", ".join([str(i) for i in range(len(choices))])

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
    return full_question

def convert_data():
    llava_data = []
    
    print(f"Reading from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            
            # Use ast.literal_eval because your data looks like a Python list string
            # e.g. ["record_id", 123, ...]
            try:
                item = ast.literal_eval(line.strip())
            except Exception as e:
                print(f"Skipping line {i} (parse error): {e}")
                continue

            # Skip header row if it exists
            if item[0] == "record_id":
                continue

            # === MAPPING YOUR FIELDS ===
            # [record_id, vid_id, vid_filename, perspective, q_body, opt0, opt1, opt2, opt3, answer]
            # 0          1       2             3            4       5     6     7     8     9
            
            record_id = str(item[0])
            video_filename = item[2]
            question = item[4]
            choices = [item[5], item[6], item[7], item[8]] # Options 0-3
            correct_answer_idx = item[9]  # Integer answer (e.g., 2)

            # Construct the prompt
            conversation_text = format_prompt(question, choices)

            # Create LLaVA training entry
            entry = {
                "id": record_id,
                "video": f"{video_filename}",  # Just the filename; loader appends folder path
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<video>\n{conversation_text}"
                    },
                    {
                        "from": "gpt",
                        "value": str(correct_answer_idx)
                    }
                ]
            }
            llava_data.append(entry)

    print(f"Converted {len(llava_data)} samples.")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(llava_data, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_data()