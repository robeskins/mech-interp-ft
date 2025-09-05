import random
from transformers import AutoTokenizer
from pathlib import Path
from utils import is_single_token, compare_token_count, generate_csvs

def get_mult_plus_add_combos(tokenizer):
    combos = []
    for a in range(2, 10):           
        for b in range(2, 10):    
            for c in range(10, 100):
                result = a * b + c
                if is_single_token(str(result), tokenizer):
                    combos.append((a, b, c, result))
    combos.sort(key=lambda x: x[3], reverse=True)
    combos = [(a, b, c) for a, b, c, _ in combos]
    random.shuffle(combos)
    return combos

def corrupt_result(a: int, b: int, tokenizer: AutoTokenizer):
    while True:
        c = random.randint(1, 100)
        answer_original = a * b
        answer_corrupted = a * b + c
        if c != 0 and compare_token_count(str(answer_original), str(answer_corrupted), tokenizer):
            return c

def create_mult_plus_add_prompts(prefix_prompt: str, total_examples: int, tokenizer: AutoTokenizer):
    combos = get_mult_plus_add_combos(tokenizer)
    prompts = []
    counter = 0

    for a, b, c in combos:
        prompt = prefix_prompt + f'{a}*{b}+{c}='
        corrupted_c = corrupt_result(a, b, tokenizer)
        corrupted_prompt = prefix_prompt + f'{a}*{b}+{corrupted_c}='
        answer = a * b + c

        length_check = compare_token_count(prompt, corrupted_prompt, tokenizer)
        if not length_check:
            continue  # skip if token counts mismatch

        prompts.append([prompt, corrupted_prompt, answer])
        counter += 1
        if counter >= total_examples:
            break

    if counter < total_examples:
        raise ValueError("Not enough examples for dataset")

    random.shuffle(prompts)
    return prompts

total_examples = 4000
prefix_prompt = 'Solve the following and respond with only the final answer:'
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")

prompts = create_mult_plus_add_prompts(prefix_prompt, total_examples, tokenizer)

generate_csvs(
    prompts=prompts,
    split_percent=0.9,
    results_dir='../pythia-1.4B/test_set_0/MultPlusAdd/datasets_csv'
)