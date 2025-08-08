import random
from transformers import AutoTokenizer
from pathlib import Path
from utils import is_single_token, compare_token_count, generate_csvs

def digit_sum(num: int) -> int:
    return sum(int(d) for d in str(num))

def get_valid_combos(tokenizer):
    valid_nums = []

    for a in range(10, 10000):
        if is_single_token(str(digit_sum(a)), tokenizer):
            valid_nums.append(a)
    random.shuffle(valid_nums)
    return valid_nums

def create_prompts(prefix_prompt: str, total_examples: int, tokenizer: AutoTokenizer):
    nums = get_valid_combos(tokenizer)
    prompts = []
    div_counter = 0

    for num in nums:
        prompt = prefix_prompt + f'Sum digits: {num} ='
        c = num + random.randint(1, 10)
        corrupted_prompt = prefix_prompt + f'Sum digits: {c} ='
        length_check = compare_token_count(prompt, corrupted_prompt, tokenizer)
        if not length_check:
            continue

        answer = digit_sum(num)
        
        if not length_check:
            raise ValueError("Token count different from prompt and corrupted!")
        prompts.append([prompt, corrupted_prompt, answer])
        div_counter +=1 
        if div_counter > total_examples:
            break
        print(div_counter)
    if div_counter < total_examples:
        raise ValueError("Not enough examples for dataset")

    return prompts

total_examples = 4000
prefix_prompt = 'Solve the following and respond with only the final answer:'
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
prompts = create_prompts(prefix_prompt, total_examples, tokenizer)

generate_csvs(prompts = prompts,
              split_percent = 0.9,
              results_dir = '../pythia-1.4B/sequential_dataset/DigitSum')