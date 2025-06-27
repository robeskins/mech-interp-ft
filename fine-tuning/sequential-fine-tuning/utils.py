from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, PeftModel, LoraConfig, PeftConfig, TaskType
from datasets import load_dataset, Dataset
from pathlib import Path
import os
from transformers.trainer_utils import get_last_checkpoint
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  

class SequentialLoRATrainer:
    def __init__(self, 
                 model_name : str, 
                 scratch_cache_dir: str,
                 lora_config : LoraConfig):
        
        self.scratch_cache_dir = scratch_cache_dir
        self.lora_config = lora_config
        self.lastest_checkpoint_path = None
        
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=os.path.join(scratch_cache_dir, "hub"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.path.join(scratch_cache_dir, "hub"))
        print(f'Loaded {model_name}')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    @staticmethod 
    def find_task_paths(tasks_directory: str) -> list[dict]:
        base_folder = Path(tasks_directory)        
        task_folders = [f for f in base_folder.iterdir() if f.is_dir()]       

        file_paths = []
        for task_folder in task_folders:
            task_name = task_folder.name 
            train_path = None
            test_path = None

            files = [f for f in task_folder.iterdir() if f.is_file()]

            for file in files:
                if 'train' in file.name:
                    train_path = file
                elif 'test' in file.name:
                    test_path = file

            if train_path is None and test_path is None:
                raise ValueError("Both train_path and test_path are None. Please unsure that files are there!")
            
            file_paths.append({
                'task_name': str(task_name),
                'train_path': str(train_path),
                'test_path': str(test_path)
            })
        return file_paths
    
    def load_and_preprocess_data(self, 
                                 train_file: str, 
                                 validation_file: str) -> Dataset: 
        data_files = {
            'train': train_file,
            'validation': validation_file
        }
        dataset = load_dataset('json', data_files=data_files)

        def preprocess_function(examples):
            max_length = 32

            inputs = examples['input']
            outputs = [str(o) for o in examples['output']]

            prompts = [f"{inp}\n" for inp in inputs]
            full_texts = [prompt + out for prompt, out in zip(prompts, outputs)]

            tokenized_full = self.tokenizer(full_texts, truncation=True, padding='max_length', max_length=max_length)

            tokenized_prompt = self.tokenizer(prompts, truncation=True, padding='max_length', max_length=max_length)

            labels = []
            for i in range(len(full_texts)):

                prompt_len = len(self.tokenizer.encode(prompts[i], truncation=True, max_length=max_length))

                label = [-100] * prompt_len + tokenized_full['input_ids'][i][prompt_len:]

                label = label[:max_length]

                if len(label) < max_length:
                    label += [-100] * (max_length - len(label))
                labels.append(label)


            tokenized_full['labels'] = labels
            return tokenized_full

        tokenized_datasets = dataset.map(preprocess_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(['input', 'output', 'instruction'])
        return tokenized_datasets
    
    def load_latest_adapter(self, 
                             is_first: bool):
        if is_first:
            lora_model = get_peft_model(self.base_model, self.lora_config)
        else:
            # peft_config = PeftConfig.from_pretrained(
            #                                         self.lastest_checkpoint_path, 
            #                                         is_trainable=True
            #                                         )
            # lora_model = get_peft_model(self.base_model, peft_config)
            # lora_model.load_adapter(
            #     self.lastest_checkpoint_path, 
            #     adapter_name="default", 
            #     is_trainable=True
            # )
            # lora_model.set_adapter("default")

            lora_model = PeftModel.from_pretrained(
                self.base_model,                 
                self.lastest_checkpoint_path,     
                is_trainable=True   
            )

        lora_model.print_trainable_parameters()
        return lora_model
    
    def train_task(self, 
                   lora_model: PeftModel,
                   dataset: Dataset, 
                   training_args: TrainingArguments,
                   output_dir: str
                   ):

        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=self.tokenizer
        )
        trainer.train()

        self.lastest_checkpoint_path = get_last_checkpoint(output_dir)
        print(f'Latest checkpoint file path:{self.lastest_checkpoint_path}')
        self.base_model = lora_model.unload() 

    def run_sequential_finetuning(self,
                                  tasks_directory: str):
            
        task_file_paths = self.find_task_paths(tasks_directory)

        gradient_accumulation_steps=4
        per_device_train_batch_size=8
        effective_batch_size = gradient_accumulation_steps * per_device_train_batch_size
        num_saves_per_epoch = 5
        num_epochs = 2

        for i,task in enumerate(task_file_paths):
            output_dir = f'./adapter_checkpoints/{task['task_name']}/'
            dataset = self.load_and_preprocess_data(task['train_path'], task['test_path'])
            train_dataset_size = len(dataset['train'])
            steps_per_epoch = math.ceil(train_dataset_size / effective_batch_size)
            calculated_save_steps = math.ceil(steps_per_epoch / num_saves_per_epoch)

            training_args = TrainingArguments(
                                                output_dir=output_dir,
                                                per_device_train_batch_size=per_device_train_batch_size,
                                                weight_decay=0.01,
                                                logging_dir='./logs',
                                                logging_steps=10,
                                                save_steps=200, 
                                                save_strategy="steps",
                                                fp16=True,
                                                gradient_accumulation_steps=gradient_accumulation_steps,
                                                report_to="none",
                                                learning_rate=3e-4,
                                                max_steps = 300, 
                                                lr_scheduler_type="constant_with_warmup"
                                            )
            
            lora_model = self.load_latest_adapter(is_first=(i == 0))
            self.train_task(lora_model = lora_model,
                            dataset = dataset,
                            training_args = training_args,
                            output_dir = output_dir
                           )
            
if __name__ == '__main__':
    scratch_cache_dir = "/mnt/fast0/rje41/.cache/huggingface"    
    model_name = "EleutherAI/pythia-1.4b-deduped"

    lora_config = LoraConfig(
                            task_type=TaskType.CAUSAL_LM,   
                            inference_mode=False,          
                            r=32,  
                            lora_alpha=64,  
                            lora_dropout=0,  
                    )

    seq_trainer = SequentialLoRATrainer(model_name = model_name,
                          scratch_cache_dir = scratch_cache_dir,
                          lora_config=lora_config)
    task_dir = 'dataset'
    seq_trainer.run_sequential_finetuning(task_dir)