import random
from transformers import AutoTokenizer
from pathlib import Path
from utils import is_single_token, compare_token_count, generate_csvs
from tqdm import tqdm

def is_valid_number(num, base):
    for digit in num:
        if not digit.isdigit() or int(digit) >= base:
            return False
    return True

def base_add(num1, num2, base): #BUG only for base 10 and below
    num1 = str(num1)
    num2 = str(num2)
    if not (is_valid_number(num1, base) and is_valid_number(num2, base)):
        return False
    
    max_len = max(len(num1), len(num2))
    num1 = num1.zfill(max_len)
    num2 = num2.zfill(max_len)
    
    carry = 0
    result = []
    
    for i in range(max_len - 1, -1, -1):
        digit_sum = int(num1[i]) + int(num2[i]) + carry
        carry = digit_sum // base
        remainder = digit_sum % base  
        
        result.append(str(remainder))
    if carry > 0:
        result.append(str(carry))
    
    result.reverse()
    return ''.join(result)

def get_valid_combos(base: int, tokenizer: AutoTokenizer):
    combos = []

    for a in tqdm(range(1, 100), desc="Processing A"):
        for b in range(1, 100):
            result = base_add(a, b, base)
            base_10_result = a + b

            if base == 10:
                if (a % 10 + b % 10) < 10:
                    continue
            else:
                if result == str(base_10_result):
                    continue

            if result and is_single_token(result, tokenizer):
                combos.append((a, b))

    random.shuffle(combos)
    print(len(combos))
    return combos

def corrupt_result(a: int, b: int, base: int, tokenizer: AutoTokenizer):
    while True:
        c = random.randint(1, 100)
        if c == b:
            continue
        
        if not compare_token_count(b, c, tokenizer):
            continue
        
        sum_ac = base_add(a, c, base)
        sum_ab = base_add(a, b, base)
        
        if sum_ac and sum_ac != sum_ab:
            return c

def create_prompts(prefix_prompt: str, total_examples: int, tokenizer: AutoTokenizer, base: int):
    combos = get_valid_combos(base, tokenizer)
    prompts = []
    counter = 0
    
    for a,b in combos:
        prompt = prefix_prompt + f'{a}+{b}='
        c = corrupt_result(a, b, base,tokenizer)
        answer = base_add(a,b,base)
        corrupted_prompt = prefix_prompt + f'{a}+{c}='
        
        length_check = compare_token_count(prompt, corrupted_prompt, tokenizer)
        if not length_check:
            raise ValueError("Token count different from prompt and corrupted!")

        prompts.append([prompt, corrupted_prompt, answer])
        counter +=1 
        if counter == total_examples:
            break
    
    if counter != total_examples:
        raise ValueError("Not enough combos for total amount")
    return prompts

base = 10
total_examples = 2500
prefix_prompt = f'Solve the following in base {base} and respond with only the final answer:'
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
prompts = create_prompts(prefix_prompt, total_examples, tokenizer, base = base)

generate_csvs(prompts = prompts,
              split_percent = 0.8,
              results_dir = '/homes/rje41/mech-interp-ft/experiment1/base_tasks_2d/AddBase10/datasets_csv')
