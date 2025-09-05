import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os 
from pathlib import Path
from utils.ft_utils import setup_model_peft, convert_csv_to_json, load_and_preprocess_data
import json
from transformers import DataCollatorForSeq2Seq

def load_edges(json_path, percentage=5):
    with open(json_path, 'r') as f:
        graph_data = json.load(f)

    scored_edges = [
    (edge, abs(attrs.get("score", 0)))
    for edge, attrs in graph_data['edges'].items()
    ]

    scored_edges.sort(key=lambda x: abs(x[1]), reverse=True)

    top_n = max(1, int(len(scored_edges) * (percentage / 100)))
    top_edges = {edge for edge, score in scored_edges[:top_n]}
    return top_edges

def jaccard_similarity(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union != 0 else 0.0

def get_jaccard_similarity_from_graphs(graph_path_1, graph_path_2, percentage=5):
    set_a = load_edges(graph_path_1, percentage)
    set_b = load_edges(graph_path_2, percentage)
    jc = jaccard_similarity(set_a, set_b)
    return jc

def get_tokenized_datasets(dataset_path: str, tokenizer):
    train_file = Path(dataset_path) / 'train.csv'
    val_file = Path(dataset_path) / 'validation.csv'
    parent_path = Path(dataset_path).parent

    filename_train_jsonl, filename_test_jsonl = convert_csv_to_json(
        str(train_file), str(val_file), str(parent_path)
    )
    tokenized_datasets = load_and_preprocess_data(filename_train_jsonl, 
                                                  filename_test_jsonl, 
                                                  tokenizer,
                                                  max_length = 32)
    return tokenized_datasets


sq_main_path = '/mnt/faster0/rje41/checkpoints/results/experiment_2/forgetting_graph_set_0_lr_false'
checkpoints_path = Path('/mnt/faster0/rje41/checkpoints/experiment_2/forgetting_graph_set_0_lr_false')

dirs = [d for d in os.listdir(sq_main_path) if os.path.isdir(os.path.join(sq_main_path, d))]

experiments = {}
for exp_folder in dirs:
    exp_path = os.path.join(sq_main_path, exp_folder)
    task_subfolders = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
    task_a_folder = [d for d in task_subfolders if 'task_a' in d][0]
    task_b_folder = [d for d in task_subfolders if 'task_b' in d][0]

    task_a_folder_path = os.path.join(exp_path, task_a_folder)
    task_b_folder_path = os.path.join(exp_path, task_b_folder)
    checkpoint_0_a = [d for d in os.listdir(task_a_folder_path) if 'checkpoint-0' in d][0]
    checkpoint_270_a = [d for d in os.listdir(task_a_folder_path) if 'checkpoint-270' in d][0]

    checkpoint_0_a_path = os.path.join(task_a_folder_path, checkpoint_0_a)
    checkpoint_270_a_path = os.path.join(task_a_folder_path, checkpoint_270_a)
   
    graph_0_a_path = Path(checkpoint_0_a_path) / 'graph.json'
    graph_270_a_path = Path(checkpoint_270_a_path) / 'graph.json'

    final_path = Path(checkpoints_path) / Path(exp_path).name / 'checkpoints' / 'checkpoint-270'

    task_name = Path(exp_path).name
    parts = task_name.split('->') 
    experiments[Path(exp_path).name ] = {
        'task_0_a_graph':graph_0_a_path,
        'task_270_a_graph':graph_270_a_path,
    }

dirs = [d for d in os.listdir(sq_main_path) if os.path.isdir(os.path.join(checkpoints_path, d))]

for check_folder in dirs:
    exp_path = os.path.join(checkpoints_path, check_folder)
    dataset_locations = Path(exp_path) / 'task_info.json'
    checkpoint_folder = [os.path.join(exp_path, d) 
                          for d in os.listdir(exp_path) 
                          if os.path.isdir(os.path.join(exp_path, d)) and d.startswith('checkpoint')][0]
    
    checkpoint_0_a_path = Path(checkpoint_folder) / 'checkpoint-0'
    checkpoint_270_a_path = Path(checkpoint_folder) / 'checkpoint-270'

    with open(dataset_locations, 'r') as f:
        data = json.load(f)
    task_a_dataset = data.get('task_a') 

    if Path(exp_path).name in experiments:
        experiments[Path(exp_path).name].update({
            'dataset_path': task_a_dataset,
            'checkpoint_0_a': checkpoint_0_a_path,
            'checkpoint_270_a': checkpoint_270_a_path
        })
    else:
        experiments[Path(exp_path).name] = {
            'dataset_path': task_a_dataset,
            'checkpoint_0_a': checkpoint_0_a_path,
            'checkpoint_270_a': checkpoint_270_a_path
        }


model_cache_dir = "/mnt/faster0/rje41/.cache/huggingface"
checkpoint_paths = '/mnt/faster0/rje41/checkpoints/experiment_2/forgetting_graph_set_0_lr_false'
model_name = "EleutherAI/pythia-1.4b-deduped"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []
for exp_name, exp_data in experiments.items():
    #Start of training Task B
    model, tokenizer = setup_model_peft(
        model_name=model_name,
        scratch_cache_dir=model_cache_dir,
        lora_r=32,
        lora_alpha=64,
        load_checkpoint=True,
        latest_checkpoint_path=str(exp_data['checkpoint_0_a'])
    )
    model.to(device)
    tokenized_datasets = get_tokenized_datasets(Path(exp_data['dataset_path']) / 'datasets_csv', tokenizer)
    val_dataset = tokenized_datasets['validation']
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, collate_fn=data_collator)
    num_batches = len(val_dataloader)

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
    
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(outputs.loss.item())

    avg_loss_0 = sum(losses) / len(losses)

    #End of training Task B
    model, tokenizer = setup_model_peft(
        model_name=model_name,
        scratch_cache_dir=model_cache_dir,
        lora_r=32,
        lora_alpha=64,
        load_checkpoint=True,
        latest_checkpoint_path=str(exp_data['checkpoint_270_a'])
    )
    model.to(device)

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
    
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(outputs.loss.item())
    avg_loss_270 = sum(losses) / len(losses)
    loss_diff = avg_loss_270 - avg_loss_0
    print('Loss diff',loss_diff)

    jaccard_similarity_value = get_jaccard_similarity_from_graphs(
        exp_data['task_0_a_graph'],
        exp_data['task_270_a_graph'],
        5
    )
    print('JC', jaccard_similarity_value)

    results.append({
        'experiment': exp_name,
        'avg_loss_0': avg_loss_0,
        'avg_loss_270': avg_loss_270,
        'loss_diff': loss_diff,
        'jaccard_similarity': jaccard_similarity_value
    })