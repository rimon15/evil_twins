# Code for the paper: Prompts have evil twins

## Installation
```conda create -n evil_twins python=3.11```

```pip install -e .[all]```

## Usage
Optimize a twin hard prompt via GCG (NOTE: for models >= 3B params, need more than 24GB VRAM)

```python
from evil_twins import load_model_tokenizer, DocDataset, optim_gcg


model_name =  "teknium/OpenHermes-2.5-Mistral-7B"
model, tokenizer = load_model_tokenizer(model_name, use_flash_attn_2=False)

optim_prompt = "! " * 15
dataset = DocDataset(
  model=model,
  tokenizer=tokenizer,
  orig_prompt="Tell me a good recipe for greek salad.",
  optim_prompt=optim_prompt,
  n_docs=100,
  doc_len=32,
  gen_batch_size=50,
)

results, ids = optim_gcg(
  model=model,
  tokenizer=tokenizer,
  dataset=dataset,
  n_epochs=500,
  kl_every=1,
  log_fpath="twin_log.json",
  id_save_fpath="twin_ids.pt",
  batch_size=10,  
  top_k=256,
  gamma=0.0, 
)
```

Optimize a soft prompt

```python
from evil_twins import load_model_tokenizer, DocDataset, optim_soft
import torch


model_name =  "teknium/OpenHermes-2.5-Mistral-7B"
model, tokenizer = load_model_tokenizer(model_name, use_flash_attn_2=False)

init_toks = torch.randint(0, model.config.vocab_size, (2,))
optim_prompt = tokenizer.decode(init_toks)
dataset = DocDataset(
  model=model,
  tokenizer=tokenizer,
  orig_prompt="Tell me a good recipe for greek salad.",
  optim_prompt=optim_prompt,
  n_docs=100,
  doc_len=32,
  gen_batch_size=50,
)

res, embs = optim_soft(
  model=model,
  dataset=dataset,
  n_epochs=500,
  kl_every=1,
  learning_rate=1e-3,
  log_fpath="soft_log.json",
  emb_save_fpath="soft_embs.pt",
  batch_size=16,
)
```

# Citation
```
@misc{melamed2024prompteviltwins,
      title={Prompt have evil twins}, 
      author={Rimon Melamed and Lucas H. McCabe and Tanay Wakhare and Yejin Kim and H. Howie Huang and Enric Boix-Adsera},
      year={2024},
      eprint={2311.07064},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2311.07064}, 
}
```