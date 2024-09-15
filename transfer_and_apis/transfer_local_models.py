# %%
# %%
import sys

sys.path.append("../")

from propane_reference import MODEL_NAME_OR_PATH_TO_NAME, PROMPT_TEMPLATES

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

from glob import glob
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm
import time
import pickle


# %%
model_name = "lmsys/vicuna-13b-v1.5"  # "meta-llama/Llama-2-13b-chat-hf"  # "teknium/OpenHermes-13B"  # "teknium/OpenHermes-13B"  # "teknium/OpenHermes-2.5-Mistral-7B"  # "lmsys/vicuna-13b-v1.5"
model = AutoModelForCausalLM.from_pretrained(
  model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%
orig_res = json.load(open("vicuna_results_paper.json", "r"))
prompts = []
for id in orig_res:
  cur = orig_res[id]
  best_arbitrary = cur["best_arbitrary_optim_kl"]
  best_gpt4_warm = cur["best_gpt4_warm_optim_kl"]
  best_gpt4_warm_natural = cur["best_gpt4_warm_natural_optim_kl"]
  best_gpt4_pruned = cur["best_gpt4_warm_pruned_optim_kl"]

  best = np.argmin(
    [best_arbitrary, best_gpt4_warm, best_gpt4_warm_natural, best_gpt4_pruned]
  )
  best_kl = np.min(
    [best_arbitrary, best_gpt4_warm, best_gpt4_warm_natural, best_gpt4_pruned]
  )
  best_optim_prompt = ""
  if best == 0:
    best_optim_prompt = cur["best_arbitrary_optim_prompt"]
  elif best == 1:
    best_optim_prompt = cur["best_gpt4_warm_optim_prompt"]
  elif best == 2:
    best_optim_prompt = cur["best_gpt4_warm_natural_optim_prompt"]
  elif best == 3:
    best_optim_prompt = cur["best_gpt4_warm_pruned_optim_prompt"]

  prompts.append(
    {
      "id": id,
      "prompt": orig_res[id]["orig_prompt"],
      "best_optim": best_optim_prompt,
      "best_optim_kl": best_kl,
    }
  )

# %%
for cur in tqdm(prompts, total=len(prompts)):
  full_prompt = (
    PROMPT_TEMPLATES[MODEL_NAME_OR_PATH_TO_NAME[model_name]]["prefix"]
    + cur["prompt"]
    + PROMPT_TEMPLATES[MODEL_NAME_OR_PATH_TO_NAME[model_name]]["suffix"]
  )
  toks = tokenizer.encode(full_prompt, return_tensors="pt").to(model.device)
  output = model.generate(
    toks,
    do_sample=True,
    min_new_tokens=256,
    max_new_tokens=256,
  )[:, toks.shape[1] :]
  cur["response"] = tokenizer.decode(output[0])

  with open(f"{model_name.split('/')[-1]}_results.json", "w") as out_file:
    json.dump(prompts, out_file, indent=4, ensure_ascii=False)

# %%
