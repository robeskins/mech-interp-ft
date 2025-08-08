import random
from transformers import AutoTokenizer
from pathlib import Path
from utils import is_single_token, compare_token_count, generate_csvs

def get_valid_combos(tokenizer: AutoTokenizer):
    add_combos = [
        (a, b) for a in range(10, 100) for b in range(10, 100)
        if is_single_token(str(a + b), tokenizer)
    ]
    
    sub_combos = [
        (a, b) for a in range(10, 100) for b in range(10, 100)
        if a > b and is_single_token(str(a - b), tokenizer)
    ]

    random.shuffle(add_combos)
    random.shuffle(sub_combos)

    return add_combos, sub_combos

def corrupt_result(correct_result: int):
    while True:
        c = random.randint(10, 99)
        if c != correct_result and compare_token_count(correct_result, c, tokenizer):
            return c

def create_prompts(prefix_prompt: str, total_examples: int, tokenizer: AutoTokenizer):
    add_combos, sub_combos = get_valid_combos(tokenizer)
    prompts = []
    add_counter = 0
    sub_counter = 0
    
    for a,b in add_combos:
        prompt = prefix_prompt + f'add({a},{b})='
        c = corrupt_result(b)
        answer = a+b
        corrupted_prompt = prefix_prompt + f'add({a},{c})='
        length_check = compare_token_count(prompt, corrupted_prompt, tokenizer)
        if not length_check:
            raise ValueError("Token count different from prompt and corrupted!")

        prompts.append([prompt, corrupted_prompt, answer])
        add_counter +=1 
        if add_counter == total_examples / 2:
            break
    
    if add_counter != total_examples / 2:
        raise ValueError("Expected half of the examples to be from addition, but got a different count. May not be enough combos for the amount of exmaples.")

    for a,b in sub_combos:
        prompt = prefix_prompt + f'sub({a},{b})='
        c = corrupt_result(b)
        answer = a - b
        corrupted_prompt = prefix_prompt + f'sub({a},{c})='
        length_check = compare_token_count(prompt, corrupted_prompt, tokenizer)
        if not length_check:
            raise ValueError("Token count different from prompt and corrupted!")
        prompts.append([prompt, corrupted_prompt, answer])
        sub_counter +=1 
        if sub_counter == total_examples / 2:
            break

    if sub_counter != total_examples / 2:
        raise ValueError("Expected half of the examples to be from subtraction, but got a different count. May not be enough combos for the amount of exmaples.")

    return prompts

total_examples = 4000
prefix_prompt = 'Solve the following and respond with only the final answer:'
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
prompts = create_prompts(prefix_prompt, total_examples, tokenizer)

generate_csvs(prompts = prompts,
              split_percent = 0.9,
              results_dir = 'same_prompt_len/AddSub/datasets_csv')