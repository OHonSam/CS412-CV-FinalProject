from transformers import AutoModel, AutoTokenizer
import torch
import json
import os
from tqdm import tqdm

# model setting
model_path = 'OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448'

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
image_processor = model.get_vision_tower().image_processor

# Print VRAM usage after loading model
print(f"VRAM after loading model: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"VRAM reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

mm_llm_compress = False # use the global compress or not
if mm_llm_compress:
    model.config.mm_llm_compress = True
    model.config.llm_compress_type = "uniform0_attention"
    model.config.llm_compress_layer_list = [4, 18]
    model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
else:
    model.config.mm_llm_compress = False

# evaluation setting
max_num_frames = 512
generation_config = dict(
    do_sample=True,
    temperature=0.1,
    max_new_tokens=256,
    top_p=0.1,
    num_beams=1
)

# Load test questions (JSONL format)
test_jsonl_path = "SUTD/questions/R3_test_100.jsonl"
video_base_path = "SUTD/videos"
output_json_path = "SUTD/questions/R3_test_with_answers.json"

print(f"Loading questions from {test_jsonl_path}")

# Parse JSONL file
test_data = []
with open(test_jsonl_path, "r") as f:
    for line_num, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        # Skip header row
        if line_num == 0 and row[0] == "record_id":
            continue
        # Parse row: [record_id, vid_id, vid_filename, perspective, q_body, option0, option1, option2, option3, answer]
        test_data.append({
            "record_id": row[0],
            "vid_id": row[1],
            "vid_filename": row[2],
            "perspective": row[3],
            "question": row[4],
            "options": [row[5], row[6], row[7], row[8]],
            "answer": row[9]
        })

print(f"Total questions: {len(test_data)}")

def format_mcq_prompt(question, options):
    """Format a multiple choice question with options."""
    prompt = f"{question}\n\nOptions:\n"
    option_labels = ['A', 'B', 'C', 'D']
    for i, opt in enumerate(options):
        if opt:  # Only include non-empty options
            prompt += f"{option_labels[i]}. {opt}\n"
    prompt += "\nPlease answer with the letter (A, B, C, or D) of the correct option."
    return prompt

def parse_model_answer(output, options):
    """Parse model output to extract the selected option index."""
    output_upper = output.upper().strip()
    
    # Check for direct letter answer
    for i, letter in enumerate(['A', 'B', 'C', 'D']):
        if output_upper.startswith(letter) or f"ANSWER IS {letter}" in output_upper or f"ANSWER: {letter}" in output_upper:
            return i
    
    # Check if output contains the option text
    for i, opt in enumerate(options):
        if opt and opt.lower() in output.lower():
            return i
    
    return -1  # Could not parse

# Process each question
results = []
correct = 0
total = 0

for idx, item in enumerate(tqdm(test_data, desc="Processing videos")):
    try:
        # Construct video path
        vid_filename = item["vid_filename"]
        video_path = os.path.join(video_base_path, vid_filename)
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            item["model_answer"] = "ERROR: Video not found"
            item["model_answer_idx"] = -1
            results.append(item)
            continue
        
        # Format question with options
        question = format_mcq_prompt(item["question"], item["options"])
        
        if idx < 3:  # Print first few for debugging
            print(f"\nProcessing {vid_filename}")
            print(f"Question: {question[:200]}...")
        
        # Run inference
        output, _ = model.chat(
            video_path=video_path, 
            tokenizer=tokenizer, 
            user_prompt=question, 
            return_history=True, 
            max_num_frames=max_num_frames, 
            generation_config=generation_config
        )
        
        # Parse answer
        predicted_idx = parse_model_answer(output, item["options"])
        
        # Add model answer to the item
        item["model_output"] = output
        item["model_answer_idx"] = predicted_idx
        item["is_correct"] = (predicted_idx == item["answer"])
        results.append(item)
        
        # Update accuracy
        total += 1
        if item["is_correct"]:
            correct += 1
        
        # Print progress
        if (idx + 1) % 50 == 0:
            print(f"\nAccuracy at {idx + 1}: {correct}/{total} = {correct/total*100:.2f}%")
        
        # Clear cache periodically
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\nError processing {item.get('vid_filename', 'unknown')}: {str(e)}")
        item["model_output"] = f"ERROR: {str(e)}"
        item["model_answer_idx"] = -1
        item["is_correct"] = False
        results.append(item)

# Calculate final accuracy
final_accuracy = correct / total * 100 if total > 0 else 0

# Save results
print(f"\nSaving results to {output_json_path}")
with open(output_json_path, "w") as f:
    json.dump({
        "accuracy": final_accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }, f, indent=2, ensure_ascii=False)

# Print final statistics
print(f"\n{'='*50}")
print(f"Final Results:")
print(f"Accuracy: {correct}/{total} = {final_accuracy:.2f}%")
print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print(f"Results saved to {output_json_path}")