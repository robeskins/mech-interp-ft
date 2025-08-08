import torch, torch.nn as nn, torch.nn.init as init
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

class SoftRouter(nn.Module):
    def __init__(self, embed_dim, num_experts, temperature=1.0):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_experts)
        self.temperature = temperature
        self.num_experts = num_experts

    def forward(self, prompt_embedding, hard=False):
        logits = self.linear(prompt_embedding)
        expert_probs = torch.nn.functional.gumbel_softmax(
            logits,
            tau=self.temperature,
            hard=hard 
        )
        return expert_probs, logits

class LoRALinear(nn.Module):
    """
    Standard LoRA implementation to replace the original Linear layer.
    """
    def __init__(self, original_linear, r=4, alpha=1.0, dropout=0.0):
        super(LoRALinear, self).__init__()
        self.original_linear = original_linear  # The original Linear layer
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA parameters
        self.lora_A = nn.Parameter(torch.empty(r, original_linear.in_features))
        self.lora_B = nn.Parameter(torch.empty(original_linear.out_features, r))
        init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # Kaiming uniform initialization for lora_A
        init.zeros_(self.lora_B)                            # Initialize lora_B to zero
        
        # Freeze the parameters of the original linear layer
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        original_output = self.original_linear(x)
        x_dropped = self.dropout(x)
        lora_output = x_dropped @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        return original_output + lora_output
    
class RoutedLoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, num_experts=2, r=4, alpha=8.0):
        super().__init__()
        self.original_linear = original_linear
        self.num_experts = num_experts
        self.scaling = alpha / r
        self.expert_index = 0  # default expert index
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        for _ in range(num_experts):
            A = nn.Parameter(torch.empty(r, original_linear.in_features))
            B = nn.Parameter(torch.empty(original_linear.out_features, r))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.zeros_(B)
            self.lora_A.append(A)
            self.lora_B.append(B)
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x, expert_indices=None):
        batch, seq_len, dim = x.shape
        x_flat = x.view(batch * seq_len, dim)
        out = self.original_linear(x)

        if expert_indices is None:
            A = self.lora_A[self.expert_index]
            B = self.lora_B[self.expert_index]
            adapted = (x_flat @ A.t()) @ B.t() * self.scaling
            res = adapted.view(batch, seq_len, -1)
        else:
            adapted = torch.zeros_like(out.view(batch * seq_len, -1), device=x.device)
            for i in range(batch):
                A = self.lora_A[expert_indices[i]]
                B = self.lora_B[expert_indices[i]]
                start = i * seq_len
                end = (i + 1) * seq_len
                adapted[start:end] = (x_flat[start:end] @ A.t()) @ B.t() * self.scaling
            res = adapted.view(batch, seq_len, -1)

        return out + res
    
class MoEWithPromptRouter(nn.Module):
    def __init__(self, base_model, embed_dim, num_experts=2, temperature=1.0):
        super().__init__()
        self.model = base_model
        self.router = SoftRouter(embed_dim, num_experts, temperature)
        self.expert_distribution = None  # for debugging/logging
        self.last_logits = None  # store for load balancing loss

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        prompt_embed = hidden.mean(dim=1)   

        expert_probs, logits = self.router(prompt_embed, hard=self.training)
        self.last_logits = logits
        self.expert_distribution = expert_probs 

        expert_indices = torch.argmax(expert_probs, dim=-1)
        return self.model(input_ids, attention_mask=attention_mask, expert_indices=expert_indices)

def _replace_module(model, module_name, new_module):
    modules = module_name.split('.')
    parent = model
    for sub_name in modules[:-1]:
        parent = getattr(parent, sub_name)
    setattr(parent, modules[-1], new_module)

def set_expert_index(model, expert_index: int):
    for module in model.modules():
        if isinstance(module, RoutedLoRALinear):
            module.expert_index = expert_index

def apply_routed_circuit_lora(model, critical_layers, r=4, alpha=16.0, extra_r=8, critical_alpha=16.0, dropout=0.05):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'embed_out' in name:
                continue
            if name in critical_layers:
                routed_lora = RoutedLoRALinear(
                    original_linear=module,
                    num_experts=2,
                    r=extra_r,
                    alpha=critical_alpha
                )
                _replace_module(model, name, routed_lora)
                print(f"Replaced {name} with RoutedLoRALinear (extra_r={extra_r}, critical_alpha={critical_alpha})")
            else:
                lora = LoRALinear(
                    original_linear=module,
                    r=r,
                    alpha=alpha,
                    dropout=dropout
                )
                _replace_module(model, name, lora)
                # print(f"Replaced {name} with LoRALinear (r={r}, alpha={alpha})")
    return model

def freeze_non_critical_layers(model, critical_layers):
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True

    for module in model.modules():
        if isinstance(module, (SoftRouter)):
            for param in module.parameters():
                param.requires_grad = True

    return model
