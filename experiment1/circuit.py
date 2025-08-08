import os
from functools import partial
from utils.eap_utils import run_eap_kl, run_metrics, run_eap_kl_baseline
from pathlib import Path
from eap.graph import Graph
from eap.attribute import attribute 
import json
import gc
import torch

datasets = ['../datasets/pythia-1.4B/two-digit/AddSub', '../datasets/pythia-1.4B/two-digit/AddSubInv', '../datasets/pythia-1.4B/two-digit/CondAddSub']

model_name = 'EleutherAI/pythia-1.4b-deduped'
transformer_lens_name = "pythia-1.4B-deduped"
scratch_cache_dir = "/mnt/faster0/rje41/.cache/huggingface"
results_folder_main = 'results_pythia_1.4B_05'
dataset_loader_n = 6
percentage_prune = 0.05

for dataset in datasets:
        print(f'Processing: {dataset}')
        task_path = Path(dataset) 
        end = Path(dataset).name

        #Store dataset metrics
        results_folder = Path(results_folder_main) / task_path.name
        results_folder.mkdir(parents=True, exist_ok=True)
        (results_folder / 'graphs').mkdir(parents=True, exist_ok=True)
        (results_folder / 'metrics').mkdir(parents=True, exist_ok=True)

        dataset_path = task_path / 'datasets_csv'
        checkpoint_folder = results_folder / 'checkpoints'
        valid_file_csv = dataset_path / "validation.csv"
        graph_file = results_folder / 'graphs' / f"graph_0.json"
        
        graph, model = run_eap_kl_baseline(model_name, 
                                           transformer_lens_name,
                                           scratch_cache_dir,
                                           valid_file_csv,
                                           dataset_loader_n)
        
        metric_data, graph = run_metrics(graph,
                                  model,
                                  valid_file_csv,
                                  dataset_loader_n,
                                  percentage_prune)
        
        
        graph_file = results_folder / 'graphs' / f"graph_0.json"
        graph.to_json(graph_file)
        
        data = {}

        data[0] = metric_data
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
                                      scratch_cache_dir,
                                      valid_file_csv,
                                      dataset_loader_n)
            #Generate metrics
            metric_data, graph = run_metrics(graph,
                                  model,
                                  valid_file_csv,
                                  dataset_loader_n,
                                  percentage_prune)
            
            graph.to_json(graph_file)

            data[checkpoint_id] = metric_data

            with open(results_folder / 'metrics' /f'metrics_{checkpoint_id}.json', 'w') as f:
                json.dump(data, f, indent=4)

            del graph
            del model
            gc.collect()
            torch.cuda.empty_cache()