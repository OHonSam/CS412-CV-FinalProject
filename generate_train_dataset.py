from dotenv import load_dotenv
from google import genai
import os
import json
import time
from pydantic import BaseModel, Field
from typing import List, Dict

class Item(BaseModel):
    options: List[str] = Field(description="List of 4 options for the question in order")
    answer: int = Field(description="Correct answer index (0-3)")

class Result(BaseModel):
    items: List[Item] = Field(description="List of generated items containing options and answers")

load_dotenv()

num_generations = 20
model = "gemini-2.5-pro"
start_id = 0
videos_dir = './videos/'
train_dataset_metadata_path = f'./SUTD/dont_bother_me/train_dataset.json'
generated_dataset_path = f'./SUTD/dont_bother_me/sutd_train_gt_generated.jsonl'
header = ["record_id", "vid_id", "vid_filename", "perspective", "q_body", "q_type", "option0", "option1", "option2", "option3", "answer"]

vid_id_pos = 1
filename_pos = 2
q_body_pos = 4
q_type_pos = 5
option0_pos = 6
option1_pos = 7
option2_pos = 8
option3_pos = 9
answer_pos = 10

generated_dataset = set()
if os.path.exists(generated_dataset_path):
    with open(generated_dataset_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            data = json.loads(line)
            generated_dataset.add((data[q_body_pos], data[filename_pos]))

print("Connecting to Gemini API...")
client = genai.Client()
print("Done")

with open(train_dataset_metadata_path, 'r', encoding='utf-8') as f:
    train_dataset = json.load(f)

    total_generations = 0
    for item in train_dataset:
        question = item['question']
        for video_info in item['videos']:
            occurrences = video_info['occurrences']
            total_generations += num_generations
    print(f"Expected total generations: {total_generations}")
    
    cur = 0
    for item in train_dataset:
        question = item['question']
        question_type = item['type']
        for video_info in item['videos']:
            cur += 1

            filename = video_info['filename']
            vid_id = video_info['vid_id']
            perspective = video_info['perspective']
            occurrences = video_info['occurrences']

            if (question, filename) in generated_dataset:
                print(f"Skipping already generated question-video pair: {question} - {filename}")
                continue

            print(f"Uploading video {filename} to Gemini...")
            video_path = os.path.join(videos_dir, filename)
            video_file = client.files.upload(file=video_path)
            print("Done")

            while video_file.state.name == "PROCESSING":
                print("Video is still processing, waiting for 10 seconds...")
                time.sleep(10)
                video_file = client.files.get(name=video_file.name)

            if video_file.state.name != "ACTIVE":
                print(f"Video processing failed with state: {video_file.state.name}, exiting...")
                exit()

            answers_text = []
            for occ in occurrences:
                answer_idx = int(occ['answer'])
                answer_text = occ['options'][answer_idx]
                answers_text.append(f"\"{answer_text}\"")
            answers_text = ', '.join(answers_text)

            prompt = f"""
You are given a video, a question about the video. Your task is to generate exactly {num_generations} pair of 4 options and answer for the question.

Question: {question}

Example (a few options and answer to illustrate the format):
{json.dumps(occurrences, ensure_ascii=False, indent=2)}

Important notes:
- Think step by step about the video content to generate plausible options.
- You must generate exactly {num_generations} pair of 4 options and answer.
- The answer index must be randomly between 0 and 3.
- The options should be plausible answers to the question based on the video content.
- The options should be diverse and cover different possible answers.
- The answer should be the most accurate option based on the video content.
- The answer in the examples (which are {answers_text}) are the most accurate based on the video content, you should follow the exact same logic.
- You must verify that the answer you provide is indeed among the 4 options you generated, the other 3 options should not be the same as the answer and should be different from each other.
"""

            print("Example:")
            print(json.dumps(occurrences, ensure_ascii=False, indent=2))

            print(f"Generating content for question \"{question}\"...")
            response = client.models.generate_content(
                model=model,
                contents=[
                    video_file,
                    prompt,
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": Result.model_json_schema(),
                }
            )
            print("Done")

            result = Result.model_validate_json(response.text)
            print(f"Generated {len(result.items)} items")

            assert len(result.items) == num_generations, f"Expected {num_generations} generations, but got {len(result.items)}"

            print("Saving generated data...")      
            if not os.path.exists(generated_dataset_path):
                with open(generated_dataset_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(json.dumps(header, ensure_ascii=False) + '\n')

            for i, gen_item in enumerate(result.items):
                record_id = start_id + (cur - 1) * num_generations + i
                output_data = [
                    record_id,
                    vid_id,
                    filename,
                    perspective,
                    question,
                    question_type,
                    gen_item.options[0],
                    gen_item.options[1],
                    gen_item.options[2],
                    gen_item.options[3],
                    gen_item.answer
                ]

                with open(generated_dataset_path, 'a', encoding='utf-8') as f_out:
                    f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
            print("Done")
