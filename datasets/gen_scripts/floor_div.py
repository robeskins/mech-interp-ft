import random
from transformers import AutoTokenizer
from pathlib import Path
from utils import is_single_token, compare_token_count, generate_csvs

def get_valid_combos(tokenizer):
    div_combos = []
    for a in range(2, 100):
        for b in range(2, 100):
            if a > b:
                result = a // b
                if is_single_token(str(result), tokenizer):
                    div_combos.append((a, b, result))
    div_combos.sort(key=lambda x: x[2], reverse=True)
    div_combos = [(a, b) for a, b, _ in div_combos]
    return div_combos

def corrupt_result(a: int, b: int, tokenizer: AutoTokenizer):
    while True:
        c = random.randint(1, 100)
        answer = a // b
        answer_c = a // c
        if c != b and compare_token_count(b, c, tokenizer) and answer != answer_c and a >c:
            return c

def create_prompts(prefix_prompt: str, total_examples: int, tokenizer: AutoTokenizer, symbol:str):
    div_combos = get_valid_combos(tokenizer)
    prompts = []
    div_counter = 0

    for a,b in div_combos:
        prompt = prefix_prompt + f'{a} {symbol} {b} ='
        c = corrupt_result(a, b,tokenizer)
        answer = int(a // b)
        corrupted_prompt = prefix_prompt + f'{a} {symbol} {c} ='
        length_check = compare_token_count(prompt, corrupted_prompt, tokenizer)
        
        if not length_check:
            raise ValueError("Token count different from prompt and corrupted!")
        prompts.append([prompt, corrupted_prompt, answer])
        div_counter +=1 
        if div_counter > total_examples:
            break
    if div_counter < total_examples:
        raise ValueError("Not enough examples for dataset")
    random.shuffle(prompts)
    return prompts

total_examples = 4000
prefix_prompt = 'Solve the following and respond with only the final answer: ' 
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
prompts = create_prompts(prefix_prompt, total_examples, tokenizer,'/')

generate_csvs(prompts = prompts,
              split_percent = 0.9,
              results_dir = '/homes/rje41/mech-interp-ft/experiment2/task_set_0_spacing/FloorDiv/datasets_csv')