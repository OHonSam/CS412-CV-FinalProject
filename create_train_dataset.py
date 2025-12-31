import json

question_type = 'I' # CHANGE THIS BRO
train_file_path = './SUTD/questions/R2_train.jsonl'
test_file_path = './SUTD/questions/R2_test.jsonl'
wa_file_path = './answers/sutd_llava_next_wrong_ans.csv'
questions_path = f'./SUTD/dont_bother_me/questions_{question_type.lower()}.txt'
output_path = f'./SUTD/dont_bother_me/train_dataset_{question_type.lower()}.json'

vid_id_pos = 1
filename_pos = 2
perspective_pos = 3
q_body_pos = 4
q_type_pos = 5
option0_pos = 6
option1_pos = 7
option2_pos = 8
option3_pos = 9
answer_pos = 10

questions_list = []
with open(questions_path, 'r', encoding='utf-8') as f:
    for line in f:
        questions_list.append(line.strip())

question_to_video = {}
filename_to_vid_id = {}
filename_to_perspective = {}
with open(test_file_path, 'r', encoding='utf-8') as f:
    next(f)

    lines = f.readlines()

    for question in questions_list:
        for line in lines:
            data = json.loads(line)
            body = data[q_body_pos]
            ques_type = data[q_type_pos]
            filename = data[filename_pos]

            if ques_type == question_type and body == question:
                option0 = data[option0_pos]
                option1 = data[option1_pos]
                option2 = data[option2_pos]
                option3 = data[option3_pos]
                answer = data[answer_pos]

                if question not in question_to_video:
                    question_to_video[question] = {}

                if filename not in question_to_video[question]:
                    question_to_video[question][filename] = []

                filename_to_vid_id[filename] = data[vid_id_pos]
                filename_to_perspective[filename] = data[perspective_pos]
                question_to_video[question][filename].append({
                    'options': [option0, option1, option2, option3],
                    'answer': answer
                })

tmp = []
for question, videos in question_to_video.items():
    cur = {}
    cur['question'] = question
    cur['type'] = question_type
    sorted_videos = sorted(videos.items(), key=lambda x: len(x[1]), reverse=True)
    cur['videos'] = []
    assert len(sorted_videos) >= 5, f"Less than 5 videos for question: {question}"
    for video, details in sorted_videos[:5]:
        print(f"  Video: {video}, Number of occurrences: {len(details)}")
        cur['videos'].append({
            'vid_id': filename_to_vid_id[video],
            'filename': video,
            'perspective': filename_to_perspective[video],
            'occurrences': details
        })
    tmp.append(cur)
    print()

with open(output_path, 'w', encoding='utf-8') as f_out:
    json.dump(tmp, f_out, indent=4)
        