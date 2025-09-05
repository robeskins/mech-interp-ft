from utils.eap_utils import run_eap_kl, run_metrics
import os
from pathlib import Path
import json
from eap.graph import Graph
import gc
import torch
import json

def main(model_name: str,
        transformer_lens_name: str,
        model_cache_dir: str,
        checkpoints_cache_dir: str,
        run_name: str,
        loader_n: int = 6,
        percentage_prune: float = 0.05,
        results_output_root: str = './results'):

    checkpoint_root = Path(checkpoints_cache_dir) / run_name 
    task_folders = [p for p in checkpoint_root.iterdir() if p.is_dir()]
    results_ouput = f'{results_output_root}/{run_name}'

    for task in task_folders:
        results_path = Path(results_ouput) / task.name 
        if results_path.exists():
            print('Already processed:',results_path)
            continue
        
        task_info_file = task / 'task_info.json'
        with open(task_info_file, "r") as f:
            task_info = json.load(f)    

        task_a_path = Path(task_info['task_a'])        
        task_b_path = Path(task_info['task_b'])

        
        task_a_validation = task_a_path / 'datasets_csv' / 'validation.csv'
        task_b_validation = task_b_path / 'datasets_csv' / 'validation.csv' 
        checkpoint_folder = task / 'checkpoints'
        checkpoints = [f for f in checkpoint_folder.iterdir() if f.is_dir()]    

        for checkpoint in checkpoints:
            checkpoint_name = checkpoint.name        
            result_dir_a = Path(results_ouput) / task.name / "task_a" / checkpoint_name        
            result_dir_a.mkdir(parents=True, exist_ok=True)         
            
            # === TASK A ===        
            graph_a, model_a = run_eap_kl(
                checkpoint,        
                model_name,        
                transformer_lens_name,        
                model_cache_dir,        
                task_a_validation,        
                loader_n
            )   

            metric_data_a, graph_a = run_metrics(        
                graph_a,        
                model_a,        
                task_a_validation,        
                loader_n,        
                percentage_prune,
            )   

        
            with open(result_dir_a / "metrics.json", "w") as f:
                json.dump(metric_data_a, f, indent=2)
        
            graph_a.to_json(result_dir_a / 'graph.json')    

        
            del model_a
            del graph_a
            torch.cuda.empty_cache()
            gc.collect()    

        
            # === TASK B ===
            result_dir_b = Path(results_ouput) / task.name / "task_b" / checkpoint_name
            result_dir_b.mkdir(parents=True, exist_ok=True)

            graph_b, model_b = run_eap_kl(
                checkpoint,
                model_name,
                transformer_lens_name,
                model_cache_dir,
                task_b_validation,
                loader_n
            )
            
            metric_data_b, graph_b = run_metrics(
                graph_b,
                model_b,
                task_b_validation,
                loader_n,
                percentage_prune, 
            )   

            with open(result_dir_b / "metrics.json", "w") as f:
                json.dump(metric_data_b, f, indent=2)

            graph_b.to_json(result_dir_b / 'graph.json')    

            del model_b
            del graph_b
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == '__main__':
    main(model_name = 'EleutherAI/pythia-1.4b-deduped',
        transformer_lens_name = 'pythia-1.4B-deduped',
        model_cache_dir = "/mnt/faster0/rje41/.cache/huggingface",
        checkpoints_cache_dir = "/mnt/faster0/rje41/checkpoints/experiment_2",
        run_name = 'forgetting_graph_set_0_lr_false_spaced',
        loader_n = 6,
        percentage_prune =  0.05,
        results_output_root = '/mnt/faster0/rje41/checkpoints/results/experiment_2'
        )
    
    