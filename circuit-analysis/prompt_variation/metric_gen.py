import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from functools import partial
from utils import load_adapter_into_hooked_transformer, EAPDataset, kl_divergence, calculate_accuracy, calculate_faithfulness
from pathlib import Path
from eap.graph import Graph
from eap.attribute import attribute 
import json

SCRATCH_STORAGE = "/mnt/faster0/rje41/.cache/huggingface"   
MODEL_NAME = "EleutherAI/pythia-1.4B-deduped"
TRANSFORMER_LENS_NAME = "pythia-1.4B-deduped"
RUN_NAME = '07-07-25'
ADAPTER_PATHS = '../../fine-tuning/add_sub_nlp/checkpoints/'

percentage_prunes = [0.05]
allowed_checkpoint_ids = [500]

if __name__ == '__main__':
    master_checkpoint_folder = Path('graph_only') / RUN_NAME
    graph_paths = [f for f in master_checkpoint_folder.iterdir() if f.suffix == '.json']

    for graph_path in graph_paths:        
        parts = graph_path.stem.split('_')
        prompt_id = parts[-2]
        checkpoint_id = parts[-1]
        if int(checkpoint_id) in allowed_checkpoint_ids:
            print(f'Processing prompt_id:{prompt_id}, checkpoint_id: {checkpoint_id}')
            g = Graph.from_json(graph_path)
            adapter_path = ADAPTER_PATHS + f'prompt_template_{prompt_id}/checkpoint-{checkpoint_id}'

            model = load_adapter_into_hooked_transformer(
                adapter_path=adapter_path,
                hf_model_name=MODEL_NAME,
                translens_model_name=TRANSFORMER_LENS_NAME,
                scratch_cache_dir=SCRATCH_STORAGE,
            )

            ds = EAPDataset(f'../../fine-tuning/add_sub_nlp/datasets_csv/prompts_id_{prompt_id}/test.csv')
            dataloader = ds.to_dataloader(6)

            save_path = Path(f'metrics/07-07-25/prompt_template_{prompt_id}_{checkpoint_id}.json')
            save_path.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            for percentage in percentage_prunes:
                print(f'Percentage:{percentage}')
                g = Graph.from_json(graph_path)
                total_edges = len(g.edges)
                prune_edges = int(total_edges * percentage)
                g.apply_topn(prune_edges, absolute=True, prune=True)

                metric_fn = partial(kl_divergence, loss=False, mean=False)
                faithfulness, percentage_performance = calculate_faithfulness(model, g, dataloader, metric_fn)
                baseline_performance, circuit_performance = calculate_accuracy(model, g, dataloader)

                data[percentage] = {
                    'faithfulness': faithfulness,
                    'percentage_performance': percentage_performance,
                    'model_accuracy': baseline_performance,
                    'graph_accuracy': circuit_performance
                }

            with open(save_path, 'w') as f:
                json.dump(data, f, indent=4)

        else:
            print(f'skipping prompt_id:{prompt_id}, checkpoint_id: {checkpoint_id}')
