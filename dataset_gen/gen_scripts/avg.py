import random
from transformers import AutoTokenizer
from pathlib import Path
from utils import is_single_token, compare_token_count, generate_csvs

def get_valid_combos(tokenizer):
    combos = []
    for a in range(10, 100):    
        for b in range(10, 100): 
            if (a + b) % 2 == 0: 
                result = (a + b) // 2
                if is_single_token(str(result), tokenizer):
                    combos.append((a, b))
    random.shuffle(combos)
    return combos

def corrupt_result(a: int, b: int, tokenizer: AutoTokenizer):
    while True:
        c = random.randint(10, 100)
        if ((a + c) % 2 == 0) and (c != b) and compare_token_count(str(b), str(c), tokenizer):
            return c
        
def create_prompts(prefix_prompt: str, total_examples: int, tokenizer: AutoTokenizer):
    div_combos = get_valid_combos(tokenizer)
    prompts = []
    div_counter = 0

    for a,b in div_combos:
        prompt = prefix_prompt + f'avg({a},{b})='
        c = corrupt_result(a, b,tokenizer)
        answer = int((a + b) // 2)
        corrupted_prompt = prefix_prompt + f'avg({a},{c})='
        length_check = compare_token_count(prompt, corrupted_prompt, tokenizer)
        
        if not length_check:
            raise ValueError("Token count different from prompt and corrupted!")
        prompts.append([prompt, corrupted_prompt, answer])
        div_counter +=1 
        if div_counter > total_examples:
            break
    if div_counter < total_examples:
        raise ValueError("Not enough examples for dataset")

    return prompts

total_examples = 4000
prefix_prompt = 'Solve the following and respond with only the final answer:'
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")

prompts = create_prompts(prefix_prompt, total_examples, tokenizer)
generate_csvs(prompts = prompts,
              split_percent = 0.9,
              results_dir = '../pythia-1.4B/test_set_1/Avg/datasets_csv')
