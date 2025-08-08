import os
from functools import partial
from utils.eap_utils import run_eap_kl, run_metrics, run_eap_kl_baseline
from pathlib import Path
from eap.graph import Graph
from eap.attribute import attribute 
import json
import gc
import yaml
import argparse
import torch

def main(config: dict):
    #Load in variables
    model_name = config["model_name"]
    cache_dir = config["scratch_cache_dir"]
    loader_n = config["dataset_loader_n"]
    transformer_lens_name = config["transformer_lens_name"]
    percentage_prune = config["percentage_prune"]
    folder_paths = config["folder_paths"]

    for folder_path in folder_paths:
        print(f'Processing: {folder_path}')
        task_path = Path(folder_path) 
        dataset_path = task_path / 'datasets_csv'
        checkpoint_folder = task_path / 'checkpoints'
        results_folder = Path(*task_path.parts[-2:])
        results_folder.mkdir(parents=True, exist_ok=True)
        (results_folder / 'graphs').mkdir(parents=True, exist_ok=True)
        (results_folder / 'metrics').mkdir(parents=True, exist_ok=True)

        valid_file_csv = dataset_path / "validation.csv"
        graph_file = results_folder / 'graphs' / f"graph_0.json"
        
        graph, model = run_eap_kl_baseline(model_name, 
                                           transformer_lens_name,
                                           cache_dir,
                                           valid_file_csv,
                                           loader_n)
        
        metric_data = run_metrics(graph,
                                  percentage_prune,
                                  model,
                                  valid_file_csv,
                                  loader_n)
        
        graph_file = results_folder / 'graphs' / f"graph_0.json"
        graph.to_json(graph_file)
        
        data = {}
        checkpoint_id = 0
        data[checkpoint_id] = metric_data
        with open(results_folder / 'metrics' /f'metrics_0.json', 'w') as f:
            json.dump(data, f, indent=4)

        del graph
        del model
        gc.collect()
        torch.cuda.empty_cache()

        
        checkpoint_folders = [f for f in checkpoint_folder.iterdir() if f.is_dir()]
        for checkpoint_folder in checkpoint_folders:
            data = {}
            checkpoint_id = checkpoint_folder.name.replace('checkpoint-','')
            graph_file = results_folder / 'graphs' / f"graph_{checkpoint_id}.json"
            if graph_file.exists():
                print(f"Graph for checkpoint {checkpoint_id} already exists. Skipping...")
                continue
            
            #Generate graph
            graph, model = run_eap_kl(checkpoint_folder,
                                      model_name, 
                                      transformer_lens_name,
                                      cache_dir,
                                      valid_file_csv,
                                      loader_n)
            graph.to_json(graph_file)
            
            #Generate metrics
            metric_data = run_metrics(graph,
                                      percentage_prune,
                                      model,
                                      valid_file_csv,
                                      loader_n)

            data[checkpoint_id] = metric_data

            with open(results_folder / 'metrics' /f'metrics_{checkpoint_id}.json', 'w') as f:
                json.dump(data, f, indent=4)

            del graph
            del model
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, "r") as file:
        data = yaml.safe_load(file)

    main(data)

