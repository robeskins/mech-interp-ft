from utils.eap_utils import run_eap_kl, run_metrics, run_eap_kl_baseline
import os
from pathlib import Path
import json
from eap.graph import Graph
import gc
import torch
import json

def run_metrics_adapter(path_valid: str, 
                        task: str,
                        model_name: str,
                        transformer_lens_name: str,
                        model_cache_dir: str,
                        loader_n: str,
                        percentage_prune: str,
                        results_ouput: str):
    #Run on basemodel
    task_validation = Path(path_valid) / 'datasets_csv' / 'validation.csv'
    results_0 = Path(results_ouput) / task.name / 'checkpoint-0'

    results_0.mkdir(parents=True, exist_ok=True)
    graph, model = run_eap_kl_baseline(model_name, 
                                       transformer_lens_name,
                                       model_cache_dir,
                                       task_validation,
                                       loader_n)
        
    metric_data, graph = run_metrics(
                 graph,
                 model,
                 task_validation,
                 loader_n,
                 percentage_prune,
             )
        
    graph.to_json(results_0 / 'graph.json')
     
    with open(results_0 / "metrics.json", "w") as f:
        json.dump(metric_data, f, indent=2)

    del model
    del graph
    torch.cuda.empty_cache()
    gc.collect()
    
    #Run through checkpoints
    checkpoint_folder = task / 'checkpoints'
    checkpoints = [f for f in checkpoint_folder.iterdir() if f.is_dir()]
    for checkpoint in checkpoints:
         checkpoint_name = checkpoint.name
         result_dir = Path(results_ouput) / task.name / checkpoint_name
         result_dir.mkdir(parents=True, exist_ok=True)

         graph, model = run_eap_kl(
             checkpoint,
             model_name,
             transformer_lens_name,
             model_cache_dir,
             task_validation,
             loader_n
         )

         metric_data, graph = run_metrics(
             graph,
             model,
             task_validation,
             loader_n,
             percentage_prune,
         )

         with open(result_dir / "metrics.json", "w") as f:
             json.dump(metric_data, f, indent=2)
         graph.to_json(result_dir / 'graph.json')

         del model
         del graph
         torch.cuda.empty_cache()
         gc.collect()

def main(
        model_name: str,
        transformer_lens_name: str,
        model_cache_dir: str,
        checkpoint_cache_dir: str,
        run_name: str,
        loader_n: int,
        percentage_prune: float,
        dataset_root: str,
        results_output_root: str
    ):
    checkpoint_root = Path(checkpoint_cache_dir) / run_name 
    tasks = [p for p in checkpoint_root.iterdir() if p.is_dir()]
    results_ouput = f'{results_output_root}/{run_name}'
    for task in tasks:
        validation_path = Path(dataset_root) / task.name

        run_metrics_adapter(validation_path, 
                            task, 
                            model_name,
                            transformer_lens_name,
                            model_cache_dir,
                            loader_n,
                            percentage_prune,
                            results_ouput)

if __name__ == '__main__':
    # main(
    #     model_name = "EleutherAI/pythia-1.4b-deduped",
    #     transformer_lens_name = "pythia-1.4B-deduped",
    #     model_cache_dir = "/mnt/faster0/rje41/.cache/huggingface",
    #     checkpoint_cache_dir = "/mnt/faster0/rje41/checkpoints/experiment_1",
    #     run_name = "add_sub_2_epochs_spacing",
    #     loader_n = 6,
    #     percentage_prune = 0.05,
    #     dataset_root = 'add_sub_tasks_spaced',
    #     results_output_root = '/mnt/faster0/rje41/checkpoints/results/experiment_1'
    # )
    # main(
    #     model_name = "EleutherAI/pythia-1.4b-deduped",
    #     transformer_lens_name = "pythia-1.4B-deduped",
    #     model_cache_dir = "/mnt/faster0/rje41/.cache/huggingface",
    #     checkpoint_cache_dir = "/mnt/faster0/rje41/checkpoints/experiment_1",
    #     run_name = "add_sub_08_epochs_spacing",
    #     loader_n = 6,
    #     percentage_prune = 0.05,
    #     dataset_root = 'add_sub_tasks_spaced',
    #     results_output_root = '/mnt/faster0/rje41/checkpoints/results/experiment_1'
    # )
    main(
        model_name = "EleutherAI/pythia-1.4b-deduped",
        transformer_lens_name = "pythia-1.4B-deduped",
        model_cache_dir = "/mnt/faster0/rje41/.cache/huggingface",
        checkpoint_cache_dir = "/mnt/faster0/rje41/checkpoints/experiment_1",
        run_name = "base_2_epochs_200",
        loader_n = 6,
        percentage_prune = 0.05,
        dataset_root = 'base_tasks_2d',
        results_output_root = '/mnt/faster0/rje41/checkpoints/results/experiment_1'
    )
    main(
        model_name = "EleutherAI/pythia-1.4b-deduped",
        transformer_lens_name = "pythia-1.4B-deduped",
        model_cache_dir = "/mnt/faster0/rje41/.cache/huggingface",
        checkpoint_cache_dir = "/mnt/faster0/rje41/checkpoints/experiment_1",
        run_name = "base_08_epochs_200",
        loader_n = 6,
        percentage_prune = 0.05,
        dataset_root = 'base_tasks_2d',
        results_output_root = '/mnt/faster0/rje41/checkpoints/results/experiment_1'
    )