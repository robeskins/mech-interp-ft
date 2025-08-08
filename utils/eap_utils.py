from functools import partial
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformer_lens import HookedTransformer
import torch.nn.functional as F
from pathlib import Path

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute 
    
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_adapter_into_hooked_transformer(adapter_path, 
                                         hf_model_name, 
                                         translens_model_name, 
                                         adapter = True, 
                                         scratch_cache_dir = None):
    base_model = AutoModelForCausalLM.from_pretrained(hf_model_name, cache_dir=scratch_cache_dir)
    if adapter == True: 
        model_with_lora = PeftModel.from_pretrained(base_model, adapter_path)
        adapter_model = model_with_lora.merge_and_unload()
    else:
        adapter_model = base_model
    model = HookedTransformer.from_pretrained(model_name=translens_model_name, hf_model=adapter_model, cache_dir=scratch_cache_dir)  

    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    return model

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
        return row['clean'], row['corrupted'], row['answer']
    
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

def calculate_faithfulness(model, g, dataloader, metric_fn):
    baseline_performance = evaluate_baseline(model, dataloader, metric_fn).mean().item()
    circuit_performance = evaluate_graph(model, g, dataloader, metric_fn,skip_clean=False).mean().item()
    faithfulness = abs(baseline_performance - circuit_performance)
    percentage_performance = (1 - faithfulness / baseline_performance) * 100

    print(f"Baseline performance: {baseline_performance}")
    print(f"Circuit performance: {circuit_performance}")
    print(f"Faithfulness: {faithfulness}")
    print(f"Percentage of model performance achieved by the circuit: {percentage_performance:.2f}%")

    return baseline_performance, circuit_performance, faithfulness, percentage_performance

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
    print(f"Baseline accuracy: {baseline_accuracy}")
    print(f"Circuit accuracy: {graph_accuracy}")
    return baseline_accuracy, graph_accuracy

def run_eap_kl(checkpoint_folder, 
              model_name, 
              transformer_lens_name, 
              cache_dir,
              valid_file_csv,
              loader_n):
    
    model = load_adapter_into_hooked_transformer(
                    adapter_path = checkpoint_folder,
                    hf_model_name = model_name,
                    translens_model_name=transformer_lens_name,
                    adapter = True,
                    scratch_cache_dir=cache_dir,
                )   
    
    g = Graph.from_model(model)
    ds = EAPDataset(valid_file_csv)
    dataloader = ds.to_dataloader(loader_n)    
    attribute(model, g, dataloader, partial(kl_divergence, loss=True, mean=True), method='EAP-IG-inputs', ig_steps=5)
    return g, model

def run_eap_kl_baseline( 
              model_name, 
              transformer_lens_name, 
              cache_dir,
              valid_file_csv,
              loader_n):
    
    model = load_adapter_into_hooked_transformer(
                    adapter_path = None,
                    hf_model_name = model_name,
                    translens_model_name=transformer_lens_name,
                    adapter = False,
                    scratch_cache_dir=cache_dir,
                )   
    
    g = Graph.from_model(model)
    ds = EAPDataset(valid_file_csv)
    dataloader = ds.to_dataloader(loader_n)    
    attribute(model, g, dataloader, partial(kl_divergence, loss=True, mean=True), method='EAP-IG-inputs', ig_steps=5)    
    return g, model

def run_metrics(g: Graph,
                model: HookedTransformer,
                valid_file_csv: str,
                loader_n: int,
                percentage_prune: float = None): #Bug: should always prune
    
    ds = EAPDataset(valid_file_csv)
    dataloader = ds.to_dataloader(loader_n) 
    if percentage_prune:
        total_edges = len(g.edges)
        percent_edges = int(total_edges * percentage_prune)
        g.apply_topn(percent_edges , absolute=True, prune=True)
        print('Pruned the graph!')
    
    metric_fn = partial(kl_divergence, loss=False, mean=False)
    baseline_performance, circuit_performance, faithfulness, percentage_performance = calculate_faithfulness(model, g, dataloader, metric_fn)
    baseline_performance_acc, circuit_performance_acc = calculate_accuracy(model, g, dataloader)

    data = {
            'model_kl': baseline_performance,
            'circuit_kl': circuit_performance,
            'faithfulness_kl': faithfulness,
            'percentage_performance_kl': percentage_performance,
            'model_accuracy': baseline_performance_acc,
            'graph_accuracy': circuit_performance_acc
            }
    return data, g

