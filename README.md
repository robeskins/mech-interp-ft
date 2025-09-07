# mech-interp-ft

This repo uses the EAP-IG algorithim to investigate pre-training priors in LLMs and sequnetial fine-tuning.

## Installation

To install the required packages, create a virtual environment and run: pip install -r requirements.txt. 

There are some local packages that need to be installed. The EAP-IG package is in external packages. This should install in the requirements.txt. However, if not, there are install info in the README.md within. This requires pip install . from EAP-IG folder. 

In addition, we use a utils package for common packages across experiments. This should also install in the requirements.txt. However, if not run pip install . 

## Structure:

- `demo`
  - This provides a basic demo on how the tools in utils for the EAP-IG algorithm work. We include some sample data for this. 

- `dataset_gen`
  - Scripts for generating the datasets.

- `single_task_gen`
  - Scripts for training adapters, then running the EAP-IG on the checkpoints.
  - We include an example in here that will save checkpoints and graphs locally, but we ran on scratch storage.

- `prior_tasks` 
  - This is work for experiment 1. 
  - This is similar to `single_task_gen` but includes the prior training datasets.
  - The file paths will have to be updated to run. 
  - Also includes analysis script of the data, which needs to be imported from the link at bottom of readme.

- `sequential_fine_tuning`
  - This is the work for experiment 2. 
  - We include a demo of this that should export checkpoints and graphs locally.

- `multi_task_learning`
  - Folder includes all files and scripts related to multi-task learning. This was experimental work that didn't yield interesting results. However, we include it if anyone wants to extend on this.
- Each folder has the same workflow for training and circuit evaluation:
    - `train.py`: This script iterates through the tasks in the `prior_tasks` folder, fine-tunes an adapter for each task, and saves the trained adapter to a specified location.
    - `run_metrics.py` takes the adapter checkpoints and runs the EAP-IG algorithim to extract the circuits. It then evaluates the model and circuits. 
    - `run.sh` allows you to run the scripts sequentially on the cloud.  
    - Analysis scripts are also in there to analyse the circuit dynamics. 

Note: you will need to replace the model cache directories to your own. In addition, train.py variables need to align with run_metrics.py. 

## Data:

- The checkpoints and graph data is too large to include in the repo (~30GB). An example can be found in example_data in prior tasks and full data and results can be found here: https://drive.google.com/drive/folders/1yzWO2O7AaY1CrQtoK6489bE3TpLhoK9Q?usp=sharing 

## Important Details:

- Fine-tuning and circuit generation require a GPU with at least 15GB of ram. 
- In addition to this, the process is slow and best to leave as a running process. 
- For sequential fine-tuning of 55 tasks it took 24hours. 


## References:

[1] Michael Hanna, Sandro Pezzelle, Yonatan Belinkov. "Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms."  COLM 2024. https://github.com/hannamw/eap-ig
- Used for the EAP-IG package to find circuits. 

[2] - Xu Wang et al. “Towards Understanding Fine‑Tuning Mechanisms of LLMs via Circuit Analysis.” ICML 2025. https://github.com/Xu0615/FinetuneCircuits
- Used for evaluation metrics and fine-tune masking function. Present in the eap_utils.py and ft_utils.py files. 
- General inspiration and guidance to the EAP-IG package.