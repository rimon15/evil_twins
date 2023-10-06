# %%
from api_helpers import send_anthropic

import os
from tqdm import tqdm
import json
import numpy as np

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
# res = []
model = "anthropic.claude-3-sonnet-20240229-v1:0"  # "anthropic.claude-3-haiku-20240307-v1:0"
max_retries = 5
for cur in tqdm(prompts, total=len(prompts)):
  cur_retries = 0
  while True:
    try:
      conv = [
        {
          "role": "user",
          "text": cur["best_optim"],
        }
      ]
      cur["response"] = send_anthropic(conv, model)
      with open(f"{model}_results.json", "w") as out_file:
        json.dump(prompts, out_file, indent=4, ensure_ascii=False)
    except Exception as e:
      cur_retries += 1
      if cur_retries == max_retries:
        print(f"prompt {cur} FAILED after {max_retries}; skipping")
        break

      print(f"retry: {cur_retries}; error on prompt: {cur}; error: {e}")
      # time.sleep(60 * 3)
      continue
    break

# %%
