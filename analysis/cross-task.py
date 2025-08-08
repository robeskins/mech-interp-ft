import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from functools import partial

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformer_lens import HookedTransformer
import torch.nn.functional as F

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute 

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(adapter_path, hf_model_name, translens_model_name, scratch_cache_dir = None):
    base_model = AutoModelForCausalLM.from_pretrained(hf_model_name, cache_dir=scratch_cache_dir)
    model_with_lora = PeftModel.from_pretrained(base_model, adapter_path)
    model_with_lora = model_with_lora.merge_and_unload()
    model = HookedTransformer.from_pretrained(model_name=translens_model_name, hf_model=model_with_lora, cache_dir=scratch_cache_dir)  

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
        # print(f'Pred_str: {pred_str}, label_str: {label_str}')
        if pred_str == label_str:
            correct.append(1.0)
        else:
            correct.append(0.0)

    return torch.tensor(correct, device=device)

def calculate_accuracy(model, g, dataloader):
    baseline_accuracy = evaluate_baseline(model, dataloader, partial(exact_match_accuracy, model)).mean().item()
    graph_accuracy = evaluate_graph(model, g, dataloader, partial(exact_match_accuracy, model)).mean().item()   
    print(f"Baseline Accuracy:{baseline_accuracy}, Graph accuracy: {graph_accuracy}")
    return baseline_accuracy, graph_accuracy

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

def cross_task(adapter_path_0: str, 
               adapter_path_1: str,
               dataloader_path_0: str,
               dataloader_path_1: str,
               hf_model_name: str, 
               translens_model_name: str, 
               scratch_cache_dir: str):
    
    model_0 = load_model(
        adapter_path=adapter_path_0,
        hf_model_name=hf_model_name,
        translens_model_name=translens_model_name,
        scratch_cache_dir=scratch_cache_dir,
    )
    ds = EAPDataset(dataloader_path_0)
    dataloader_0 = ds.to_dataloader(6)
    g_0 = Graph.from_json('graph_0.json')
    g_1 = Graph.from_json('graph_1.json')

    faithfulness_g0_mo = calculate_faithfulness(model_0, g_0, dataloader_0)
    accuracy_g0_m0 = calculate_accuracy(model_0, g_0, dataloader_0)
    faithfulness_g1_mo = calculate_faithfulness(model_0, g_1, dataloader_0)
    accuracy_g1_m0 = calculate_accuracy(model_0, g_1, dataloader_0)

    model_1 = load_model(
        adapter_path=adapter_path_1,
        hf_model_name=hf_model_name,
        translens_model_name=translens_model_name,
        scratch_cache_dir=scratch_cache_dir,
    )
    ds = EAPDataset(dataloader_path_1)
    dataloader_1 = ds.to_dataloader(6)

    faithfulness_g1_m1 = calculate_faithfulness(model_0, g_0, dataloader_0)
    accuracy_g1_m1 = calculate_accuracy(model_0, g_0, dataloader_0)
    faithfulness_g0_m1 = calculate_faithfulness(model_1, g_0, dataloader_0)
    accuracy_g0_m1 = calculate_accuracy(model_1, g_0, dataloader_0)