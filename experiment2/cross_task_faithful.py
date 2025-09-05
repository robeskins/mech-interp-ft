import os
from utils.eap_utils import run_metrics, load_adapter_into_hooked_transformer
from eap.graph import Graph
import json
import gc
import torch
from pathlib import Path

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
    checkpoint_270_a = [d for d in os.listdir(task_a_folder_path) if 'checkpoint-270' in d][0]
    checkpoint_270_b = [d for d in os.listdir(task_b_folder_path) if 'checkpoint-270' in d][0]

    checkpoint_270_a_path = os.path.join(task_a_folder_path, checkpoint_270_a)
    checkpoint_270_b_path = os.path.join(task_b_folder_path, checkpoint_270_b)

    graph_a_path = Path(checkpoint_270_a_path) / 'graph.json'
    graph_b_path = Path(checkpoint_270_b_path) / 'graph.json'
    metrics_b_path = Path(checkpoint_270_b_path) / 'metrics.json'
    final_path = Path(checkpoints_path) / Path(exp_path).name / 'checkpoints' / 'checkpoint-270'

    task_name = Path(exp_path).name
    parts = task_name.split('->') 
    experiments[Path(exp_path).name ] = {
        'task_a_graph':graph_a_path,
        'task_b_graph':graph_b_path,
        'checkpoint_path':final_path,
        'task_a': parts[0],
        'task_b': parts[1]
    }

model_name = "EleutherAI/pythia-1.4b-deduped"
transformer_lens_name = "pythia-1.4B-deduped"
model_cache_dir = "/mnt/faster0/rje41/.cache/huggingface"
validation_datasets = '/homes/rje41/mech-interp-ft/experiment2/task_set_0'

all_results = {}
task_combos_set = [
    ('Sub','AddSub'),
    ('Add','Sub'),
    ('Abs','Sub'),
    ('AddSub','AddSubAlias'),
    ('FloorDiv','AddSubAlias'),
    ('AddSub','CondAddSub'),
    ('Sub','Modulo'),
    ('Abs','FloorDiv')
]

for exp_name, exp_info in experiments.items():
    task_a_graph = exp_info['task_a_graph']
    task_b_graph = exp_info['task_b_graph']
    checkpoint_path = exp_info['checkpoint_path']
    task_a = exp_info['task_a']
    task_b = exp_info['task_b']

    print(task_a, task_b)

    model = load_adapter_into_hooked_transformer(
        adapter_path=checkpoint_path,
        hf_model_name=model_name,
        translens_model_name=transformer_lens_name,
        adapter=True,
        scratch_cache_dir=model_cache_dir,
    )   

    result_a_on_b = run_metrics(
        g=Graph.from_json(task_a_graph),
        model=model,
        valid_file_csv=Path(validation_datasets) / task_b / 'datasets_csv' / 'validation.csv',
        loader_n=6,
        percentage_prune=0.05
    )

    result_b_on_b = run_metrics(
        g=Graph.from_json(task_b_graph),
        model=model,
        valid_file_csv=Path(validation_datasets) / task_b / 'datasets_csv' / 'validation.csv',
        loader_n=6,
        percentage_prune=0.05
    )

    result_b_on_a = run_metrics(
        g=Graph.from_json(task_b_graph),
        model=model,
        valid_file_csv=Path(validation_datasets) / task_a / 'datasets_csv' / 'validation.csv',
        loader_n=6,
        percentage_prune=0.05
    )

    result_a_on_a = run_metrics(
        g=Graph.from_json(task_a_graph),
        model=model,
        valid_file_csv=Path(validation_datasets) / task_a / 'datasets_csv' / 'validation.csv',
        loader_n=6,
        percentage_prune=0.05
    )

    kl_faithfulness_a_to_b = result_b_on_b[0]['circuit_kl'] / result_a_on_b[0]['circuit_kl']
    kl_faithfulness_b_to_a = result_a_on_a[0]['circuit_kl'] / result_b_on_a[0]['circuit_kl']

    acc_faithfulness_a_to_b = result_a_on_b[0]['graph_accuracy'] / result_b_on_b[0]['graph_accuracy']
    acc_faithfulness_b_to_a = result_b_on_a[0]['graph_accuracy'] / result_a_on_a[0]['graph_accuracy']

    del model
    torch.cuda.empty_cache()
    gc.collect()

    all_results[exp_name] = {
        'task_a_on_task_b': result_a_on_b[0],
        'task_b_on_task_b': result_b_on_b[0],
        'task_b_on_task_a': result_b_on_a[0],
        'task_a_on_task_a': result_a_on_a[0],
        'kl_faithfulness_a_to_b': kl_faithfulness_a_to_b,
        'kl_faithfulness_b_to_a': kl_faithfulness_b_to_a,
        'accuracy_faithfulness_a_to_b': acc_faithfulness_a_to_b,
        'accuracy_faithfulness_b_to_a': acc_faithfulness_b_to_a
    }

    with open('cross_task_faith.json', 'w') as f:
        json.dump(all_results, f, indent=4)