import json
import csv
import os 

def convert_csv_to_json(file_csv: str, percentage_split: float, output_dir: str, id) -> None:
    subfolder = os.path.join(output_dir, f"prompts_id_{id}")
    filename_train_jsonl = os.path.join(subfolder, "train.jsonl")
    filename_test_jsonl = os.path.join(subfolder, "test.jsonl")
    os.makedirs(subfolder, exist_ok=True)

    data = []
    with open(file_csv, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            data.append({"instruction": "", "input": row[0], "output": row[2]})

    split_index = int(len(data) * percentage_split)

    train_data = data[:split_index]
    test_data = data[split_index:]

    with open(filename_train_jsonl, 'w', encoding='utf-8') as f:
        for item in train_data:
            json_line = json.dumps(item)
            f.write(json_line + '\n')

    with open(filename_test_jsonl, 'w', encoding='utf-8') as f:
        for item in test_data:
            json_line = json.dumps(item)
            f.write(json_line + '\n')

    return filename_train_jsonl, filename_test_jsonl
