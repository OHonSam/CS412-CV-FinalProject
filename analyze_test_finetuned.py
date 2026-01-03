import os
import json

dataset_path = './SUTD/questions/R2_test.jsonl'
baseline_path = './answers/sutd_test_with_answers_video_chat.json'
result_path = './answers/sutd_test_with_answers_video_chat_yolo_prompting.csv'

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

id_to_question_type = {}
with open(dataset_path, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        data = json.loads(line)
        record_id = int(data[record_id_pos])
        q_type = data[q_type_pos]
        id_to_question_type[record_id] = q_type

baseline_count = {}
with open(baseline_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    results = data['results']
    for item in results:
        record_id = int(item['record_id'])
        correct = item['is_correct']
        question_type = id_to_question_type[record_id]
        if question_type not in baseline_count:
            baseline_count[question_type] = {
                "correct": 0,
                "total": 0
            }

        baseline_count[question_type]["total"] += 1
        if correct:
            baseline_count[question_type]["correct"] += 1

count_type = {}
with open(result_path, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        data = line.strip().split(',')

        id = int(data[0])
        correct = bool(data[4] == 'True')
        question_type = id_to_question_type[id]

        if question_type not in count_type:
            count_type[question_type] = {
                "correct": 0,
                "total": 0
            }

        count_type[question_type]["total"] += 1
        if correct:
            count_type[question_type]["correct"] += 1

for q_type, counts in count_type.items():
    correct = counts["correct"]
    total = counts["total"]
    baseline_accuracy = baseline_count[q_type]["correct"] / baseline_count[q_type]["total"]
    model_accuracy = correct / total
    print(f"Question Type: {q_type}, Baseline Accuracy: {baseline_accuracy * 100:.4f}, Model Accuracy: {model_accuracy * 100:.4f}, Delta: {(model_accuracy - baseline_accuracy) * 100:.4f}, Total: {total}")


    