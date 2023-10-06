# %%
import google.generativeai as genai
from google.generativeai import GenerationConfig
import numpy as np

from glob import glob
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm
import time
import pickle

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# %%
# for m in genai.list_models():
#   if "generateContent" in m.supported_generation_methods:
#     print(m.name)
model = genai.GenerativeModel("gemini-pro")
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
max_retries = 5
conf = GenerationConfig(max_output_tokens=256, candidate_count=1)
for cur in tqdm(prompts, total=len(prompts)):
  cur_retries = 0
  while True:
    try:
      response = model.generate_content(cur["best_optim"], generation_config=conf)
      # cur["response"] = response.text  # str(response.parts)
      cur["response"] = ""
      for p in response.parts:
        cur["response"] += p.text + "\n"
      with open("google_transfer_results.json", "w") as out_file:
        json.dump(prompts, out_file, indent=4, ensure_ascii=False)
      time.sleep(60 * 1)
    except Exception as e:
      cur_retries += 1
      if cur_retries == max_retries:
        print(f"prompt {cur} FAILED after {max_retries}; skipping")
        break

      print(f"retry: {cur_retries}; error on prompt: {cur}; error: {e}")
      time.sleep(60 * 3)
      continue
    break
