from utils.eap_utils import run_eap_kl, run_metrics, run_eap_kl_baseline
import os
from pathlib import Path
import json
from eap.graph import Graph
import json

model_name = 'EleutherAI/pythia-1.4b-deduped'
transformer_lens_name = "pythia-1.4B-deduped"
cache_dir = "/mnt/faster0/rje41/.cache/huggingface"
loader_n = 6
percentage_prune = 0.05

tasks_single = ['tasks/add_sub', 'tasks/sequence']
task_a_b = 'tasks/MultiTask/add_seq'

def process_task_checkpoints(task_path: Path, checkpoints, validation_csv: Path, save_base_path: Path = None):
    base_path = save_base_path if save_base_path else task_path
    print(validation_csv)

    graph, model = run_eap_kl_baseline(model_name, 
                                       transformer_lens_name,
                                       cache_dir,
                                       validation_csv,
                                       loader_n)
    
    result_dir = base_path / 'checkpoint-0'
    result_dir.mkdir(parents=True, exist_ok=True)

    metric_data, graph = run_metrics(
            graph,
            percentage_prune,
            model,
            validation_csv,
            loader_n
        )
    
    with open(result_dir / "metrics.json", "w") as f:
        json.dump(metric_data, f, indent=2)
    graph.to_json(result_dir / 'graph.json')

    for checkpoint in checkpoints:
        checkpoint_name = checkpoint.name
        graph, model = run_eap_kl(
            checkpoint,
            model_name,
            transformer_lens_name,
            cache_dir,
            validation_csv,
            loader_n
        )
        metric_data = run_metrics(
            graph,
            percentage_prune,
            model,
            validation_csv,
            loader_n
        )

        result_dir = base_path / checkpoint_name
        result_dir.mkdir(parents=True, exist_ok=True)

        with open(result_dir / "metrics.json", "w") as f:
            json.dump(metric_data, f, indent=2)
        graph.to_json(result_dir / 'graph.json')

def main():
    for task_str in tasks_single:
        task_path = Path(task_str)
        checkpoint_folder = task_path / 'checkpoints'
        validation_csv = task_path / 'datasets_csv' / 'validation.csv'

        if not checkpoint_folder.exists():
            print(f"Warning: Checkpoint folder not found for {task_path}")
            continue

        checkpoints = [f for f in checkpoint_folder.iterdir() if f.is_dir()]
        process_task_checkpoints(task_path, checkpoints, validation_csv)

    multitask_path = Path(task_a_b)
    checkpoint_folder = multitask_path / 'checkpoints'
    validation_csv = multitask_path / 'datasets_csv' / 'validation.csv'

    if not checkpoint_folder.exists():
        print(f"Warning: Checkpoint folder not found for {multitask_path}")
        return

    multitask_checkpoints = [f for f in checkpoint_folder.iterdir() if f.is_dir()]

    for task_str in tasks_single:
        task_path = Path(task_str)
        validation_csv_single = task_path / 'datasets_csv' / 'validation.csv'
        save_dir = multitask_path / task_path.name 
        process_task_checkpoints(task_path, multitask_checkpoints, validation_csv_single, save_base_path=save_dir)

    process_task_checkpoints(multitask_path, multitask_checkpoints, validation_csv)

if __name__ == "__main__":
    main()
