import random
from transformers import AutoTokenizer
from pathlib import Path
from utils import is_single_token, compare_token_count, generate_csvs

def get_valid_combos(tokenizer: AutoTokenizer):
    combos = [
        (a, b) for a in range(10, 100) for b in range(10, 100)
        if is_single_token(str(abs(a - b)), tokenizer)
    ]
    random.shuffle(combos)

    return combos

def corrupt_result(correct_result: int):
    while True:
        c = random.randint(10, 100)
        if c != correct_result and compare_token_count(correct_result, c, tokenizer):
            return c

def create_prompts(prefix_prompt: str, total_examples: int, tokenizer: AutoTokenizer):
    combos = get_valid_combos(tokenizer)
    prompts = []
    counter = 0
    
    for a,b in combos:
        prompt = prefix_prompt + f'{a}~{b}='
        answer = abs(a-b)
        c = corrupt_result(answer)
        corrupted_prompt = prefix_prompt + f'{a}~{c}='
        length_check = compare_token_count(prompt, corrupted_prompt, tokenizer)
        if not length_check:
            raise ValueError("Token count different from prompt and corrupted!")

        prompts.append([prompt, corrupted_prompt, answer])
        counter +=1 
        if counter == total_examples:
            break

    return prompts

total_examples = 4000
prefix_prompt = 'Solve the following and respond with only the final answer:'
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
prompts = create_prompts(prefix_prompt, total_examples, tokenizer)

generate_csvs(prompts = prompts,
              split_percent = 0.9,
              results_dir = '../Abs/datasets_csv')