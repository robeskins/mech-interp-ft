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
                        results_name: str,
                        model_name: str,
                        transformer_lens_name: str,
                        cache_dir: str,
                        loader_n: str,
                        percentage_prune: str,
                        results_ouput: str):
    #Run on basemodel
    task_validation = Path(path_valid) / 'datasets_csv' / 'validation.csv'
    results_0 = Path(results_ouput) / task.name / results_name / 'checkpoint-0'

    results_0.mkdir(parents=True, exist_ok=True)
    graph, model = run_eap_kl_baseline(model_name, 
                                       transformer_lens_name,
                                       cache_dir,
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
         result_dir = Path(results_ouput) / task.name / results_name / checkpoint_name
         result_dir.mkdir(parents=True, exist_ok=True)

         graph, model = run_eap_kl(
             checkpoint,
             model_name,
             transformer_lens_name,
             cache_dir,
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


def main(model_name: str,
         transformer_lens_name: str,
         model_cache_dir: str,
         checkpoints_cache_dir: str,
         run_name: str,
         loader_n: int,
         percentage_prune: int):
    
    checkpoints_cache_dir = Path(checkpoints_cache_dir)
    checkpoints_cache_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(checkpoints_cache_dir) / run_name
    task_folders = [p for p in results_root.iterdir() if p.is_dir()]
    results_ouput = f'./results/{run_name}' 

    for task in task_folders:
        task_info_file = task / 'task_info.json'
        with open(task_info_file, "r") as f:
            task_info = json.load(f)    


        run_metrics_adapter(task_info['task_validation_path'], 
                            task,
                            'task_single', 
                            model_name,
                            transformer_lens_name,
                            model_cache_dir,
                            loader_n,
                            percentage_prune,
                            results_ouput)

        if 'task_a_validation_path' in task_info:
            run_metrics_adapter(task_info['task_a_validation_path'], 
                                task,
                                'task_a', 
                                model_name,
                                transformer_lens_name,
                                model_cache_dir,
                                loader_n,
                                percentage_prune,
                                results_ouput)

        if 'task_b_validation_path' in task_info:
            run_metrics_adapter(task_info['task_b_validation_path'], 
                                task,
                                'task_b', 
                                model_name,
                                transformer_lens_name,
                                model_cache_dir,
                                loader_n,
                                percentage_prune,
                                results_ouput)

if __name__ == '__main__':
    main(model_name = 'EleutherAI/pythia-1.4b-deduped',
         transformer_lens_name = 'pythia-1.4B-deduped',
         model_cache_dir = '/mnt/faster0/rje41/.cache/huggingface',
         checkpoints_cache_dir = "/mnt/faster0/rje41/checkpoints/experiment_3",
         run_name = 'task_set_1_half_true',
         loader_n = 6,
         percentage_prune = 0.05)
    
    main(model_name = 'EleutherAI/pythia-1.4b-deduped',
         transformer_lens_name = 'pythia-1.4B-deduped',
         model_cache_dir = '/mnt/faster0/rje41/.cache/huggingface',
         checkpoints_cache_dir = "/mnt/faster0/rje41/checkpoints/experiment_3",
         run_name = 'task_set_1_half_false',
         loader_n = 6,
         percentage_prune = 0.05)