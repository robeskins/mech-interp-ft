# mech-interp-ft

This repo uses the EAP-IG algorithim to investigate pre-training priors in LLMs and sequnetial fine-tuning.

## Installation

To install the required packages, create a virtual environment and run: pip install -r requirements.txt. 

There are some local packages that need to be installed. The EAP-IG package is in external packages. This should install in the requirements.txt. However, if not, there are install info in the README.md within. This requires pip install . from the folder. 

In addition, we use a utils package for common packages across experiments. This should also install in the requirements.txt. 

## Structure:

- `dataset_gen` folder includes all the scripts needed to reproduce the datasets. 

- `prior_tasks` folder includes all files and scripts related to experiment 1 that are used to study pre-training priors in LLMs. This includes:

- Datasets for both the AddSub and AddBase tasks
- `train.py`: This script iterates through the tasks in the `prior_tasks` folder, fine-tunes an adapter for each task, and saves the trained adapter to a specified location.
- `run_metrics.py` takes the adapter checkpoints and runs the EAP-IG algorithim to extract the circuits. It then evaluates the model and circuits. 
- `run.sh` allows you to run the scripts sequentially on the cloud.  
- Analysis scripts are also in there to analyse the circuit dynamics. 

Note: you will need to replace the cache directories to yours. In addition, train.py variables need to align with run_metrics.py. 

`sequential_fine_tuning` folder includes all files and scripts related to experiment 2 that are used to study sequential fine-tuning in LLMs. This follows the same setup as previous:

- `train.py`: trains the adapters sequentially, this takes all the permutations or you can specify the combinations. This stores the adapter checkpoints in the desired location, alongside the file_paths for the tasks in the task_info.json.
- `run_metrics.py` takes the adapter checkpoints and runs the EAP-IG algorithim to extract the circuits. It then evaluates the model and circuits on the tasks specified in the task_info.json

`multi_task_learning` folder includes all files and scripts related to multi task learning. This was experimental work that didn't yield intersting results. However, we include if anyone wants to extend on this. 

## Data:

The checkpoints and graph data is too large to include in the repo (~30GB). An example can be found in example_data in prior tasks and full data and results can be found here: https://drive.google.com/drive/folders/1yzWO2O7AaY1CrQtoK6489bE3TpLhoK9Q?usp=sharing 

## Important Details:
File paths will have to be changed in the files to align with your local set-up

Fine-tuning and circuit generation require a GPU with at least 15GB of ram. In addition to this, the process is slow and best to leave as a running process. For sequential fine-tuning of 55 tasks ran for 24hours. 


## References:

[1] Michael Hanna, Sandro Pezzelle, Yonatan Belinkov. "Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms."  COLM 2024. https://github.com/hannamw/eap-ig
- Used for the EAP-IG package to find circuits. 

[2] - Xu Wang et al. “Towards Understanding Fine‑Tuning Mechanisms of LLMs via Circuit Analysis.” ICML 2025. https://github.com/Xu0615/FinetuneCircuits
- Used for evaluation metrics and fine-tune masking function. Present in the eap_utils.py and ft_utils.py files. 
- General inspiration and guidance to the EAP-IG package.