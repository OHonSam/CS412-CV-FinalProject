import json

question_types = ['F', 'I']
output_path = "./SUTD/dont_bother_me/train_dataset.json"

merged_data = []
for question_type in question_types:
    input_path = f'./SUTD/dont_bother_me/train_dataset_{question_type.lower()}.json'

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        merged_data.extend(data)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=4)
