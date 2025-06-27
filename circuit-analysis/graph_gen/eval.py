from functools import partial
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
import torch.nn.functional as F
from peft import PeftModel
from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute 
from functools import partial
import os
from pathlib import Path
import yaml
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row['clean'], row['corrupted'], row['label']
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def kl_divergence(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=True):
    logits = get_logit_positions(logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    probs = torch.softmax(logits, dim=-1)
    clean_probs = torch.softmax(clean_logits, dim=-1)

    results = F.kl_div(probs.log(), clean_probs.log(), log_target=True, reduction='none').mean(-1)
    return results.mean() if mean else results


def load_model(adapter_path: str, hf_model_name: str, translens_model_name: str, scratch_cache_dir: str = None):
    base_model = AutoModelForCausalLM.from_pretrained(hf_model_name, cache_dir=scratch_cache_dir)
    model_with_lora = PeftModel.from_pretrained(base_model, adapter_path)
    model_with_lora = model_with_lora.merge_and_unload()
    model = HookedTransformer.from_pretrained(model_name=translens_model_name, hf_model=model_with_lora, cache_dir=scratch_cache_dir)  

    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    return model

def calculate_faithfulness(model, g, dataloader, metric_fn):
    baseline_performance = evaluate_baseline(model, dataloader, metric_fn).mean().item()
    circuit_performance = evaluate_graph(model, g, dataloader, metric_fn).mean().item()
    faithfulness = abs(baseline_performance - circuit_performance)
    percentage_performance = (1 - faithfulness / baseline_performance) * 100
    return faithfulness, percentage_performance

def exact_match_accuracy(model, logits, corrupted_logits, input_lengths, labels):
    batch_size = logits.size(0)
    device = logits.device
    positions = input_lengths - 1

    last_logits = logits[torch.arange(batch_size), positions, :]
    predicted_tokens = last_logits.argmax(dim=-1)
    predicted_strings = [model.to_string(token.item()).strip() for token in predicted_tokens]

    labels_strings = []
    for i in range(batch_size):
        lab = labels[i]
        if isinstance(lab, torch.Tensor):
            lab = lab.item()
        labels_strings.append(str(lab).strip())

    correct = []
    for pred_str, label_str in zip(predicted_strings, labels_strings):
        if pred_str == label_str:
            correct.append(1.0)
        else:
            correct.append(0.0)

    return torch.tensor(correct, device=device)

def calculate_accuracy(model, g, dataloader):
    baseline_accuracy = evaluate_baseline(model, dataloader, partial(exact_match_accuracy, model)).mean().item()
    graph_accuracy = evaluate_graph(model, g, dataloader, partial(exact_match_accuracy, model)).mean().item()   
    return baseline_accuracy, graph_accuracy

def evaluate_checkpoint(scratch_cache_dir: str, 
                        hf_model_name: str, 
                        translens_model_name: str, 
                        adapter_path: str,
                        percentage_prune: float,
                        circuit_analyis_path: str):
    model = load_model(
        adapter_path=adapter_path,
        hf_model_name=hf_model_name,
        translens_model_name=translens_model_name,
        scratch_cache_dir=scratch_cache_dir,
    )

    ds = EAPDataset(circuit_analyis_path)
    dataloader = ds.to_dataloader(6)

    g = Graph.from_model(model)
    attribute(model, g, dataloader, partial(kl_divergence, loss=True, mean=True), method='EAP-IG', ig_steps=5)

    total_edges = len(g.edges)
    five_percent_edges = int(total_edges * percentage_prune)
    g.apply_topn(five_percent_edges , absolute=True)
    baseline_performance, circuit_performance = calculate_accuracy(model, g, dataloader)
    faithfulness, percentage_performance = calculate_faithfulness(model, g, dataloader, partial(kl_divergence, loss=False, mean=False))
    return g, faithfulness, percentage_performance, baseline_performance, circuit_performance

def main(config_file: str):
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    checkpoint_folder = config["checkpoint_folder"]
    graph_output_folder = config["graph_output_folder"]
    results_path = config["results_path"]
    hf_model_name = config["hf_model_name"]
    translens_model_name = config["translens_model_name"]
    scratch_cache_dir = config["scratch_cache_dir"]
    circuit_analyis_path = config["circuit_analyis_path"]

    Path(graph_output_folder).mkdir(parents=True, exist_ok=True)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(results_path).exists():
        try:
            with open(results_path, "r") as f:
                results_data = json.load(f)
            if not isinstance(results_data, list):
                raise ValueError("Expected a list in the results JSON.")
        except (json.JSONDecodeError, ValueError):
            results_data = []
    else:
        results_data = []

    folder_path = Path(checkpoint_folder)
    adapter_folders = [f for f in folder_path.iterdir() if f.is_dir()]

    for adapter in adapter_folders:
        print(f"Evaluating: {adapter.name}")

        g, faithfulness, percentage_performance, baseline_performance, circuit_performance = evaluate_checkpoint(
            scratch_cache_dir = scratch_cache_dir,
            hf_model_name = hf_model_name,
            translens_model_name = translens_model_name,
            adapter_path= adapter,
            percentage_prune = 0.05,
            circuit_analyis_path = circuit_analyis_path
        )

        graph_path = Path(graph_output_folder) / adapter.name
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        g.to_json(str(graph_path) + '.json')

        new_result = {
            'adapter': adapter.name,
            'faithfulness': faithfulness,
            'percentage_performance': percentage_performance,
            'baseline_performance_exact': baseline_performance,
            'circuit_performance_exact': circuit_performance
        }
        results_data.append(new_result)
        print(results_data)

        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=4)

if __name__ == '__main__':
    main('eval.yaml')