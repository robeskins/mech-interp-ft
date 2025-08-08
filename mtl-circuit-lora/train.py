from mtl_circuit_lora import SoftRouter, LoRALinear, RoutedLoRALinear, apply_routed_circuit_lora, freeze_non_critical_layers, MoEWithPromptRouter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, torch.nn as nn
from transformers import Trainer
from transformers import TrainingArguments
from utils.ft_utils import setup_model_peft, convert_csv_to_json, load_and_preprocess_data
from pathlib import Path
from torch.nn import functional as F

def get_tokenized_datasets(dataset_path: str, tokenizer):
    train_file = Path(dataset_path) / 'train.csv'
    val_file = Path(dataset_path) / 'validation.csv'
    parent_path = Path(dataset_path).parent

    filename_train_jsonl, filename_test_jsonl = convert_csv_to_json(
        str(train_file), str(val_file), str(parent_path)
    )
    tokenized_datasets = load_and_preprocess_data(filename_train_jsonl, 
                                                  filename_test_jsonl, 
                                                  tokenizer,
                                                  max_length = 64)
    return tokenized_datasets


class MoETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask', None))
        logits = outputs.logits 

        labels = inputs['input_ids']
        loss_lm = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        expert_probs = getattr(model, 'expert_distribution', None)
        if expert_probs is not None:
            importance = expert_probs.mean(dim=0)
            load_balancing_loss = model.router.num_experts * (importance ** 2).sum()
        else:
            load_balancing_loss = 0.0

        lambda_load_balancing = 0.01
        loss = loss_lm + lambda_load_balancing * load_balancing_loss

        return (loss, outputs) if return_outputs else loss
    
model_name = "EleutherAI/pythia-410m-deduped"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# critical_layers = [
#     "gpt_neox.layers.20.mlp.dense_4h_to_h",
#     "gpt_neox.layers.22.attention.dense" 
# ]
patched_model = apply_routed_circuit_lora(base_model, critical_layers=[], r= 1, alpha = 1)
model = freeze_non_critical_layers(patched_model, [])
embed_dim = getattr(base_model.config, 'n_embd', None) or getattr(base_model.config, 'hidden_size', None)
moe_model = MoEWithPromptRouter(model, embed_dim, num_experts=2)

training_args = TrainingArguments(
    output_dir='./lora_gpt_results/r32a64',  
    num_train_epochs=2,                          
    per_device_train_batch_size=8,                 
    warmup_steps=25,                              
    weight_decay=0.01,                            
    logging_dir='./circuit_weighted_lora_logs',   
    logging_steps=1,                             
    save_steps=1,                                
    save_strategy="no",                        
    save_total_limit=10,                           
    fp16=True,                                   
    gradient_accumulation_steps=4,               
    report_to="none",                             
    learning_rate=3e-4,                           
)
tokenized_datasets = get_tokenized_datasets('datasets_csv', tokenizer)
trainer = MoETrainer(
    model=moe_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
)
trainer.train()
trainer.evaluate()