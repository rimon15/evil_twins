# Code for the paper: Prompts have evil twins (EMNLP 2024)

## Installation

### From github

```pip install evil_twins@git+https://github.com/rimon15/evil_twins```

### From source
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
```bibtex
@inproceedings{melamed-etal-2024-prompts,
    title = "Prompts have evil twins",
    author = "Melamed, Rimon  and
      McCabe, Lucas  and
      Wakhare, Tanay  and
      Kim, Yejin  and
      Huang, H. Howie  and
      Boix-Adser{\`a}, Enric",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.4",
    pages = "46--74",
    abstract = "We discover that many natural-language prompts can be replaced by corresponding prompts that are unintelligible to humans but that provably elicit similar behavior in language models. We call these prompts {``}evil twins{''} because they are obfuscated and uninterpretable (evil), but at the same time mimic the functionality of the original natural-language prompts (twins). Remarkably, evil twins transfer between models. We find these prompts by solving a maximum-likelihood problem which has applications of independent interest.",
}
```
