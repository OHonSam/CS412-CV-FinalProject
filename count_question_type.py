import os
import json

train_dataset_path = "./SUTD/questions/R2_train.jsonl"

# Question Type: U, Count: 35178
# Question Type: A, Count: 11094
# Question Type: C, Count: 2824
# Question Type: R, Count: 2470
# Question Type: I, Count: 2380
# Question Type: F, Count: 2514

question_type_name = {
    "U": "Basic Understanding",
    "A": "Attribution",
    "I": "Introspection",
    "C": "Counterfactual Inference",
    "R": "Reverse Reasoning",
    "F": "Event Forecasting",
}

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

question_type_count = {}
with open(train_dataset_path, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        data = json.loads(line)
        q_type = data[q_type_pos]
        if q_type not in question_type_count:
            question_type_count[q_type] = 0
        question_type_count[q_type] += 1

total = sum(question_type_count.values())
for q_type, count in question_type_count.items():
    print(f"Question Type: {q_type}, Question Type Name: {question_type_name[q_type]}, Count: {count}")
print(f"Total Questions: {total}")