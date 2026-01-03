import json

question_type = 'I'  # CHANGE THIS BRO

train_file_path = './SUTD/questions/R2_train.jsonl'
test_file_path = './SUTD/questions/R2_test.jsonl'
wa_file_path = './answers/sutd_llava_next_wrong_ans.csv'
output_path = 'unique_questions.txt'


unique_questions = {}
q_body_pos = 4
q_type_pos = 5

try:
    count_train = 0
    count_test = 0
    with open(train_file_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            data = json.loads(line)
            
            id = data[0]
            body = data[q_body_pos]
            ques_type = data[q_type_pos]

            if ques_type == question_type:
                count_train += 1
                
                if body not in unique_questions:
                    unique_questions[body] = [0, 0, 0]
                unique_questions[body][0] += 1

    id_to_question = {}
    all_ids = set()
    with open(test_file_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:

            data = json.loads(line)

            id = int(data[0])
            body = data[q_body_pos]
            ques_type = data[q_type_pos]
            
            if ques_type == question_type:
                all_ids.add(id)
                count_test += 1
                if body not in unique_questions:
                    unique_questions[body] = [0, 0, 0]
                unique_questions[body][1] += 1

            if id not in id_to_question:
                id_to_question[id] = body

    with open(wa_file_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:

            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            id = int(parts[0])

            if id in id_to_question and id in all_ids:
                question = id_to_question[id]
                if question in unique_questions:
                    unique_questions[question][2] += 1
            
    with open(output_path, 'w', encoding='utf-8') as f_out:
        sorted_questions = sorted(unique_questions.items(), key=lambda x: 1 - ((x[1][2] / x[1][1]) if x[1][1] != 0 else 0), reverse=True)
        for question, counts in sorted_questions:
            f_out.write(f"Train: {counts[0]}, {counts[0] / count_train * 100:.2f}%\t\tTest: {counts[1]}, {counts[1] / count_test * 100:.2f}%\t\tRatio: {counts[0] / (counts[1] if counts[1] != 0 else 1):.2f}\t\tWA: {counts[2]}\t\tAccuracy: {(1 - ((counts[2] / counts[1]) if counts[1] != 0 else 0)) * 100:.2f}%\t\tQuestion: {question}\n")

except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")