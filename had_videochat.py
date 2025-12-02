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

# Load test questions
test_json_path = "/datastore/clc_hcmus/ZaAIC/CS412-CV-FinalProject/HAD/questions/test.json"
video_base_path = "/datastore/clc_hcmus/ZaAIC/CS412-CV-FinalProject/HAD/videos/test"
output_json_path = "/datastore/clc_hcmus/ZaAIC/CS412-CV-FinalProject/HAD/questions/test_with_answers.json"

print(f"Loading questions from {test_json_path}")
with open(test_json_path, "r") as f:
    test_data = json.load(f)

print(f"Total questions: {len(test_data)}")

# Process each question
results = []
for idx, item in enumerate(tqdm(test_data, desc="Processing videos")):
    try:
        # Construct video path
        video_id = item.get("video_id")
        video_path = os.path.join(video_base_path, video_id)
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            item["model_answer"] = "ERROR: Video not found"
            results.append(item)
            continue
        
        # Get question
        qa_pair = item.get("QA", "")
        question = qa_pair.get("q", "")
        print(f"\nProcessing {video_id}: {question}")
        # Run inference
        output, _ = model.chat(
            video_path=video_path, 
            tokenizer=tokenizer, 
            user_prompt=question, 
            return_history=True, 
            max_num_frames=max_num_frames, 
            generation_config=generation_config
        )
        
        # Add model answer to the item
        item["model_answer"] = output
        results.append(item)
        
        # Clear cache periodically
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
            print(f"\nVRAM usage at {idx + 1}: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f"\nError processing {video_id}: {str(e)}")
        item["model_answer"] = f"ERROR: {str(e)}"
        results.append(item)

# Save results
print(f"\nSaving results to {output_json_path}")
with open(output_json_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Print final VRAM usage
print(f"\nFinal VRAM usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print(f"Completed! Results saved to {output_json_path}")