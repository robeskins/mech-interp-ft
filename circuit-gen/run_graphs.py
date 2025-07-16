import os
from functools import partial
from utils import load_adapter_into_hooked_transformer, EAPDataset, kl_divergence, calculate_accuracy, calculate_faithfulness
from pathlib import Path
from eap.graph import Graph
from eap.attribute import attribute 
import json
import yaml
import argparse

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

        valid_file_csv = dataset_path / "validation.csv"
        ds = EAPDataset(valid_file_csv)
        dataloader = ds.to_dataloader(loader_n)    

        #Iterate over checkpoints
        data = {}
        checkpoint_folders = [f for f in checkpoint_folder.iterdir() if f.is_dir()]
        for checkpoint_folder in checkpoint_folders:
            checkpoint_id = checkpoint_folder.name.replace('checkpoint-','')
            graph_file = results_folder / 'graphs' / f"graph_{checkpoint_id}.json"

            if graph_file.exists():
                print(f"Graph for checkpoint {checkpoint_id} already exists. Skipping...")
                continue
            
            model = load_adapter_into_hooked_transformer(
                    adapter_path = checkpoint_folder,
                    hf_model_name = model_name,
                    translens_model_name=transformer_lens_name,
                    scratch_cache_dir=cache_dir,
                )   
            #Generate graph
            g = Graph.from_model(model)
            attribute(model, g, dataloader, partial(kl_divergence, loss=True, mean=True), method='EAP-IG-inputs', ig_steps=5)

            total_edges = len(g.edges)

            five_percent_edges = int(total_edges * percentage_prune)
            g.apply_topn(five_percent_edges , absolute=True, prune=True)
            g.to_json(results_folder / 'graphs' / f"graph_{checkpoint_id}.json")

            metric_fn = partial(kl_divergence, loss=False, mean=False)
            faithfulness, percentage_performance = calculate_faithfulness(model, g, dataloader, metric_fn)
            baseline_performance, circuit_performance = calculate_accuracy(model, g, dataloader)

            data[checkpoint_id] = {
                                    'faithfulness': faithfulness,
                                    'percentage_performance': percentage_performance,
                                    'model_accuracy': baseline_performance,
                                    'graph_accuracy': circuit_performance
                                    }

        with open(results_folder / 'metrics.json', 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, "r") as file:
        data = yaml.safe_load(file)

    main(data)

