import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from utils.eap_utils import run_metrics, load_adapter_into_hooked_transformer
from eap.graph import Graph
import json
import gc
import torch

#Replace with local files and adapter paths
paths = ['/mnt/faster0/rje41/checkpoints/results/experiment_1/base_4_epochs_small_ds/AddBase8/checkpoint-200/graph.json', 
        '/mnt/faster0/rje41/checkpoints/results/experiment_1/base_4_epochs_small_ds/AddBase9/checkpoint-200/graph.json', 
        '/mnt/faster0/rje41/checkpoints/results/experiment_1/base_4_epochs_small_ds/AddBase10/checkpoint-200/graph.json']

adapter_paths = ['/mnt/faster0/rje41/checkpoints/experiment_1/base_4_epochs_small_ds/AddBase8/checkpoints/checkpoint-200',
                '/mnt/faster0/rje41/checkpoints/experiment_1/base_4_epochs_small_ds/AddBase9/checkpoints/checkpoint-200',
                '/mnt/faster0/rje41/checkpoints/experiment_1/base_4_epochs_small_ds/AddBase10/checkpoints/checkpoint-200']

validation_dataset_paths = ['/homes/rje41/mech-interp-ft/experiment1/base_tasks_2d/AddBase8/datasets_csv/validation.csv',
                           '/homes/rje41/mech-interp-ft/experiment1/base_tasks_2d/AddBase9/datasets_csv/validation.csv',
                           '/homes/rje41/mech-interp-ft/experiment1/base_tasks_2d/AddBase10/datasets_csv/validation.csv']

model_name = "EleutherAI/pythia-1.4b-deduped"
transformer_lens_name = "pythia-1.4B-deduped"
model_cache_dir = "/mnt/faster0/rje41/.cache/huggingface"
all_results = []

for i in range(len(paths)):
    model = load_adapter_into_hooked_transformer(
        adapter_path=adapter_paths[i],
        hf_model_name=model_name,
        translens_model_name=transformer_lens_name,
        adapter=True,
        scratch_cache_dir=model_cache_dir,
    )   

    percentages = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.3, 0.5]
    graph = Graph.from_json(paths[i])
    run_results = []

    for percentage in percentages:
        result = run_metrics(
            g=graph,
            model=model,
            valid_file_csv=validation_dataset_paths[i],
            loader_n=6,
            percentage_prune=percentage
        )
        run_results.append({
            "percentage_prune": percentage,
            "metrics": result[0]
        })

    all_results.append({
        "adapter_path": adapter_paths[i],
        "graph_path": paths[i],
        "validation_dataset": validation_dataset_paths[i],
        "results": run_results
    })
    del model
    del graph

output_path = "pruning_results.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"Saved results to {output_path}")
