import os
import json
import random

# Get list of video files in videos/test
test_videos_dir = "HAD/videos/test"
video_files = os.listdir(test_videos_dir)
# Remove file extensions to get video IDs
video_ids = set([f for f in video_files])

print(f"Found {len(video_ids)} videos in {test_videos_dir}")
print(video_ids)
# Load the HAD-instruct-test.json file
with open("HAD/questions/HAD-instruct-test.json", "r") as f:
    all_questions = json.load(f)

print(f"Total questions in HAD-instruct-test.json: {len(all_questions)}")

# Filter questions that have video_id in the video list
filtered_questions = [q for q in all_questions if q.get("video_id") in video_ids]

print(f"Questions matching available videos: {len(filtered_questions)}")

# Sample 1000 questions (or all if less than 1000)
sample_size = min(1000, len(filtered_questions))
sampled_questions = random.sample(filtered_questions, sample_size)

print(f"Sampled {sample_size} questions")

# Save the sampled questions
output_file = "HAD-instruct-test-sampled-1000.json"
with open(output_file, "w") as f:
    json.dump(sampled_questions, f, indent=2)

print(f"Saved sampled questions to {output_file}")