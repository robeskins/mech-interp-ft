import random
from transformers import AutoTokenizer
from pathlib import Path
from utils import is_single_token, compare_token_count, generate_csvs

def digit_sum(num: int) -> int:
    return sum(int(d) for d in str(num))

def get_valid_combos(tokenizer):
    valid_pairs = []
    for a in range(10, 100):
        for b in range(10, 100):
            ds = digit_sum(a) + digit_sum(b)
            if is_single_token(str(ds), tokenizer):
                valid_pairs.append((a, b))
    random.shuffle(valid_pairs)
    print(len(valid_pairs))
    return valid_pairs

def create_prompts(prefix_prompt: str, total_examples: int, tokenizer: AutoTokenizer, symbol: str):
    pairs = get_valid_combos(tokenizer)
    prompts = []
    count = 0

    for a, b in pairs:
        prompt = prefix_prompt + f'{a}{symbol}{b}='
        while True:
            c = random.randint(10, 100)
            if c != b and compare_token_count(b, c, tokenizer):
                break
        
        corrupted_prompt = prefix_prompt + f'{a}{symbol}{c}='
        length_check = compare_token_count(prompt, corrupted_prompt, tokenizer)
        if not length_check:
            continue
        
        answer = digit_sum(a) + digit_sum(b)

        prompts.append([prompt, corrupted_prompt, answer])
        count += 1
        if count >= total_examples:
            break

    if count < total_examples:
        raise ValueError("Not enough examples for dataset")

    return prompts

total_examples = 4000
prefix_prompt = 'Solve the following and respond with only the final answer:'
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
prompts = create_prompts(prefix_prompt, total_examples, tokenizer, '◇')

generate_csvs(prompts = prompts,
              split_percent = 0.9,
              results_dir = '../pythia-1.4B/sequential_dataset_same_prompt/DigitSum/datasets_csv')