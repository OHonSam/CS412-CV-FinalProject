import os
import json
import time
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from google.genai import types
import base64
import argparse

models = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
]

model = "gemini-3-pro-preview"
test_dataset_path = "./SUTD/questions/R2_test.jsonl"
videos_dir = './videos/'
predictions_output_path = f"./SUTD/dont_bother_me/gemini_predictions_{model}.json"

record_id_pos = 0
vid_id_pos = 1
filename_pos = 2
q_body_pos = 4
q_type_pos = 5
option0_pos = 6
option1_pos = 7
option2_pos = 8
option3_pos = 9
answer_pos = 10

load_dotenv()

class Result(BaseModel):
    predict: int = Field(description="Predicted answer option index (0-3)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    
    if args.model not in models:
        raise ValueError(f"Model {args.model} is not supported. Choose from {models}")

    model = args.model
    predictions_output_path = f"./SUTD/dont_bother_me/gemini_predictions_{model}.json"
    
    print("Loading existing predictions...")
    predictions = {}
    count_total = 0
    count_correct = 0
    if os.path.exists(predictions_output_path):
        with open(predictions_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                predictions[item['id']] = {
                    "pd": item['pd'],
                    "gt": item['gt'],
                }

                if item['pd'] == item['gt']:
                    count_correct += 1
                count_total += 1
    print("Done")

    print("Connecting to Gemini API...")
    client = genai.Client()
    print("Done")

    with open(test_dataset_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            data = json.loads(line)
            
            record_id = data[record_id_pos]

            if record_id in predictions:
                print(f"Skipping record_id {record_id} as it is already predicted")
                continue

            print(f"Processing record_id {record_id}...")

            question = data[q_body_pos]
            filename = data[filename_pos]
            vid_id = data[vid_id_pos]

            video_path = os.path.join(videos_dir, filename)
            option0 = data[option0_pos]
            option1 = data[option1_pos]
            option2 = data[option2_pos]
            option3 = data[option3_pos]
            answer = data[answer_pos]

            video_file = None
            if not model.startswith("gemini-3"):  
                print(f"Uploading video {filename}...")
                video_path = os.path.join(videos_dir, filename)
                video_file = client.files.upload(file=video_path)
                while video_file.state.name == "PROCESSING":
                    print("Video is still processing, waiting for 2 seconds...")
                    time.sleep(2)
                    video_file = client.files.get(name=video_file.name)
                print("Done")

            prompt = f"""
    You are given a video, a question about the video and four answer options.
    Your task is to select the most correct answer option based on the content of the video.

    Video: {filename}
    Question: {question}
    Options:
    - 0: {option0}
    - 1: {option1}
    - 2: {option2}
    - 3: {option3}

    Important: Only provide the index (0-3) of the most correct answer option as your response.
    """.strip()

            print(f"Generating content for record_id {record_id}...")
            response = None
            if model.startswith("gemini-3"):
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Content(
                            parts=[
                                types.Part(text=prompt),
                                types.Part(
                                    inline_data=types.Blob(
                                        mime_type="video/mp4",
                                        data=base64.b64encode(
                                            open(video_path, "rb").read()
                                        ).decode("utf-8")
                                    )
                                )
                            ]
                        )
                    ],
                    config={
                        "tools": [
                            {"google_search": {}},
                        ],
                        "response_mime_type": "application/json",
                        "response_json_schema": Result.model_json_schema(),
                    }
                )
                print("Done")
            else:
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

            if not model.startswith("gemini-3"):
                print("Deleting uploaded video to save space...")
                client.files.delete(name=video_file.name)
                print("Done")
            
            result = Result.model_validate_json(response.text)
            predictions[record_id] = {
                "pd": result.predict,
                "gt": answer,
            }

            if result.predict == answer:
                count_correct += 1
            count_total += 1

            with open(predictions_output_path, 'w', encoding='utf-8') as f_out:
                output_data = []
                for rid, res in predictions.items():
                    output_data.append({
                        "id": rid,
                        "pd": res['pd'],
                        "gt": res['gt'],
                    })
                json.dump(output_data, f_out, ensure_ascii=False, indent=2)
            
            print(f"Saved prediction for record_id {record_id}")
            print(f"Current accuracy: {count_correct}/{count_total} = {count_correct / count_total:.4f}")
            print("-" * 50)

    print("All done")