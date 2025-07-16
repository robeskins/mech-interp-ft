import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from functools import partial
from utils import load_adapter_into_hooked_transformer, EAPDataset, kl_divergence
from pathlib import Path
from eap.graph import Graph
from eap.attribute import attribute 

SCRATCH_STORAGE = "/mnt/faster0/rje41/.cache/huggingface"   
MODEL_NAME = "EleutherAI/pythia-1.4B-deduped"
TRANSFORMER_LENS_NAME = "pythia-1.4B-deduped"
RUN_NAME = '07-07-25_no_prefix'

if __name__ == '__main__':
    master_checkpoint_folder = Path('../../fine-tuning/add_sub_nlp/checkpoints_no_prefix/')
    prompt_folders = [f for f in master_checkpoint_folder.iterdir() if f.is_dir()]

    for prompt_folder in prompt_folders:
        checkpoint_folders = [f for f in prompt_folder.iterdir() if f.is_dir()]
        for checkpoint_folder in checkpoint_folders:
            prompt_template_id = prompt_folder.name.replace('prompt_template_','')
            checkpoint_id = checkpoint_folder.name.replace('checkpoint-','')

            results_folder = f'graph_only/{RUN_NAME}/'
            results_folder = Path(results_folder)
            results_folder.mkdir(parents=True, exist_ok=True)   
            file_path = results_folder / f"graph_{prompt_folder.name}_{checkpoint_id}.json"
            if file_path.exists():
                print(f'file_path: {file_path} already exists, skipping..')

            else:
                print(f'Processing: {file_path}')
                model = load_adapter_into_hooked_transformer(
                        adapter_path=checkpoint_folder,
                        hf_model_name=MODEL_NAME,
                        translens_model_name=TRANSFORMER_LENS_NAME,
                        scratch_cache_dir=SCRATCH_STORAGE,
                    )   

                ds = EAPDataset(f'../../fine-tuning/add_sub_nlp/datasets_csv_no_prefix/prompts_id_{prompt_template_id}/test.csv')
                dataloader = ds.to_dataloader(6)    

                g = Graph.from_model(model)
                attribute(model, g, dataloader, partial(kl_divergence, loss=True, mean=True), method='EAP-IG-inputs', ig_steps=5)
                g.to_json(results_folder / f"graph_{prompt_folder.name}_{checkpoint_id}.json")
