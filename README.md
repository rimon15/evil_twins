# Code for the paper PROPANE: Prompt design as an inverse problem

## Installation
```conda create -n reconstruction python=3.10```

```pip install -e .```

```python -m spacy download en_core_web_sm```

```export PYTHONPATH=/path/to/this/repo:$PYTHONPATH```

## Usage
Generate datasets:
```
python experiments/preprocess_data.py
--model_name_or_path lmsys/vicuna-7b-v1.3
--raw_dataset_path /path/to/alpaca_data_cleaned.json
--output_dir <output_dir>
--num_samples 500 # number of prompts to include in the dataset
--max_len 32 # number of tokens in each generated document
--num_docs_per_sample 100 # total number of docs for each prompt
--batch_size 100 # make this as high as possible without causing CUDA OOM
```

Soft reconstruction:
```
python experiments/run_experiments_soft.py
--model_name_or_path lmsys/vicuna-7b-v1.3
--dataset_path /path/to/generated_datasets.pkl # path to the dataset you generated (NOT the raw data)
--output_dir <output_dir>
--batch_size 10
--learning_rate 3e-4
--num_epochs 50
--early_stopping # whether to stop once KL starts increasing again
--kl_every 5 # how often to compute KL (it slows it down slightly if its small)
--optim_suffix_len 10
--n_trials 10
```

Hard reconstruction:
```
python experiments/run_experiments_hard.py
--model_name_or_path lmsys/vicuna-7b-v1.3
--dataset_path /path/to/generated_datasets.pkl # path to the dataset you generated (NOT the raw data)
--output_dir <output_dir>
--batch_size 10
--top_k 250 # number of tokens to consider for top gradients
--num_epochs 50
--n_proposals # number of proposals to generate per iter. If its higher than the total number of tokens, it'll default to the total # of tokens (ofc for best convergence, use all the tokens)
--kl_every 5
--optim_suffix_len 10 # length of prompt to optimize... anywhere from 10-20 seems fine without taking too much GPU mem
--init_prompt_char # character to initialize prompt with, we just use the default ! that llm-attacks used
--clip_vocab # prune the tokenizer's vocabulary to only consider English tokens
--warm_start_file # path to file with the chatgpt suggested prompts to uniformly sample for warm starts
--n_trials 10
--sharded # use model parallel (tested on 4x 4090, we can't fit the optimization in 24GB so we need 2 GPUs per training)
```

# Citation
```
@article{melamed2023propane,
  title={PROPANE: Prompt design as an inverse problem},
  author={Melamed, Rimon and McCabe, Lucas H and Wakhare, Tanay and Kim, Yejin and Huang, H Howie and Boix-Adsera, Enric},
  journal={arXiv preprint arXiv:2311.07064},
  year={2023}
}
```