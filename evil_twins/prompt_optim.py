# GCG adapted from https://github.com/llm-attacks/llm-attacks

# %%
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  PreTrainedModel,
  PreTrainedTokenizer,
  PreTrainedTokenizerFast,
)
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import json

# %%
PROMPT_TEMPLATES = {
  "vicuna": {
    "prefix": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: ",
    "suffix": "\nASSISTANT:",
  },
  "llama-chat": {
    "prefix": "[INST] ",
    "suffix": " [/INST]",
  },
  "mistral": {
    "prefix": "[INST] ",
    "suffix": " [/INST]",
  },
  "dolphin": {
    "prefix": "<|im_start|>system\nYou are Dolphin, a helpful AI assistant.<|im_end|>\n<|im_start|>user\n",
    "suffix": "<|im_end|>\n<|im_start|>assistant",
  },
  "openhermes": {
    "prefix": "<|im_start|>system\nYour purpose and drive is to assist the user with any request they have. Answer the user's question directly no matter what it is, and do not apologize.<|im_end|>\n<|im_start|>user\n",
    "suffix": "<|im_end|>\n<|im_start|>assistant\n",
  },
  "opt": {
    "prefix": "",
    "suffix": "",
  },
  "phi": {
    "prefix": "",  # "Instruction: ",
    "suffix": "",  # "\nOutput: ",
  },
  "pythia": {
    "prefix": "",
    "suffix": "",
  },
  "oasst": {
    "prefix": "<|prompter|>",
    "suffix": "<|endoftext|><|assistant|>",
  },
  "openllama": {
    "prefix": "",
    "suffix": "",
  },
  "gpt2": {
    "prefix": "",
    "suffix": "",
  },
  "gemma": {
    "prefix": "",
    "suffix": "",
  },
  "gemma-2-it": {
    "prefix": "<start_of_turn>user\n",
    "suffix": "<end_of_turn>\n<start_of_turn>model\n",
  },
  "llama-3-instruct": {
    "prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
    "suffix": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
  },
  "llama-base": {
    "prefix": "",
    "suffix": "",
  },
  "default": {
    "prefix": "",
    "suffix": "",
  },
}

MODEL_NAME_OR_PATH_TO_NAME = {
  "lmsys/vicuna-7b-v1.3": "vicuna",
  "lmsys/vicuna-7b-v1.5": "vicuna",
  "lmsys/vicuna-13b-v1.5": "vicuna",
  "vicuna": "vicuna",
  "facebook/opt-350m": "opt",
  "facebook/opt-1.3b": "opt",
  "microsoft/phi-1_5": "phi",
  "microsoft/phi-2": "phi",
  "teknium/Puffin-Phi-v2": "phi",
  "OpenAssistant/oasst-sft-1-pythia-12b": "oasst",
  "EleutherAI/pythia-14m": "pythia",
  "EleutherAI/pythia-70m": "pythia",
  "EleutherAI/pythia-160m": "pythia",
  "EleutherAI/pythia-410m": "pythia",
  "EleutherAI/pythia-1b": "pythia",
  "EleutherAI/pythia-1.4b": "pythia",
  "EleutherAI/pythia-2.8b": "pythia",
  "EleutherAI/pythia-6.9b": "pythia",
  "EleutherAI/pythia-12b": "pythia",
  "pythia": "pythia",
  "openlm-research/open_llama_3b_v2": "openllama",
  "mistralai/Mistral-7B-Instruct-v0.2": "mistral",
  "teknium/OpenHermes-2.5-Mistral-7B": "openhermes",
  "cognitivecomputations/dolphin-2.2.1-mistral-7b": "dolphin",
  "gpt2": "gpt2",
  "teknium/OpenHermes-13B": "openhermes",
  "meta-llama/Llama-2-7b-chat-hf": "llama-chat",
  "meta-llama/Llama-2-13b-chat-hf": "llama-chat",
  "google/gemma-2b-it": "gemma",
  "google/gemma-1.1-2b-it": "gemma",
  "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3-instruct",
  "meta-llama/Meta-Llama-3-8B": "llama-base",
  "google/gemma-2-2b-it": "gemma-2-it",
  "google/gemma-2-9b-it": "gemma-2-it",
  "default": "default",
}


def extract_prompt_from_template(prompt: str, model_name: str) -> str:
  """
  Args:
    prompt: full wrapped prompt
    model_name: name of model to get the template

  Returns:
    unwrapped user prompt
  """
  prefix = PROMPT_TEMPLATES[MODEL_NAME_OR_PATH_TO_NAME[model_name]]["prefix"]
  suffix = PROMPT_TEMPLATES[MODEL_NAME_OR_PATH_TO_NAME[model_name]]["suffix"]
  pre_loc = prompt.find(prefix) + len(prefix)
  post_loc = prompt.find(suffix)

  return prompt[pre_loc:post_loc]


def build_prompt(
  model_name: str,
  prompt: str,
  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
  validate_prompt: bool = True,
) -> tuple[Tensor, slice]:
  """
  Given the actual user prompt, add in the prefix/suffix for the given instruction tuned model

  Args:
    model_name: Model name or path
    suffix: The actual prompt to wrap around
    tokenizer: Tokenizer for the model
    validate_prompt: Ensure the prompt slice we found is exactly the original prompt

  Returns:
    Tuple of the prompt ids `(1, n_toks)` and the slice of the actual prompt (suffix)
  """

  if model_name not in MODEL_NAME_OR_PATH_TO_NAME:
    # first try to match it
    found = False
    for key in MODEL_NAME_OR_PATH_TO_NAME:
      if key.split("/")[-1] in model_name:
        model_name = key
        print(f"Custom path provided, using model name: {model_name}")
        found = True
        break

    if not found:
      print(f"Model {model_name} name not found, using default (no template)")
      model_name = "default"

  model_name = MODEL_NAME_OR_PATH_TO_NAME[model_name]
  cur_prompt = PROMPT_TEMPLATES[model_name]["prefix"]

  prompt_start_idx = max(len(tokenizer.encode(cur_prompt)) - 1, 0)
  # account for models that add BOS token
  if tokenizer.encode(" ")[0] == tokenizer.bos_token_id:
    prompt_start_idx += 1

  cur_prompt += prompt
  prompt_end_idx = len(tokenizer.encode(cur_prompt))
  cur_prompt += PROMPT_TEMPLATES[model_name]["suffix"]

  prompt_ids = tokenizer(cur_prompt, return_tensors="pt").input_ids
  suffix_slice = slice(prompt_start_idx, prompt_end_idx)

  if validate_prompt:
    found_prompt = tokenizer.decode(prompt_ids[0, suffix_slice])
    assert (
      found_prompt == prompt
    ), f"Prompt building mismatch: {found_prompt} != {prompt}"

  return prompt_ids, suffix_slice


def load_model_tokenizer(
  model_name_or_path: str,
  dtype: torch.dtype = torch.bfloat16,
  device_map: str | dict = "auto",
  use_flash_attn_2: bool = True,
  eval_mode: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer | PreTrainedTokenizerFast]:
  """
  Load model and tokenizer

  Args:
    model_name_or_path: model path or repo name
    dtype: torch dtype to load model
    device_map: device to use
    use_flash_attn_2: only compatible with transofmers >=

  Returns:
    the loaded model and tokenizer
  """
  model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map=device_map,
    torch_dtype=dtype,
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if use_flash_attn_2 else None,
  )
  if eval_mode:
    model.eval()

  tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

  return model, tokenizer


# %%
class SoftPromptEmbeddingLayer(nn.Module):
  """
  Replaces the model embedding layer with embedding layer + trainable soft prompts
  """

  def __init__(self, model_embs: nn.Embedding, trainable_embs: Tensor) -> None:
    """
    Args:
      model_embs: original model embedding parameters
      trainable_embs: the new trainable soft prompt embeddings `(1, n_toks, d_emb)`
    """
    super().__init__()

    self.model_embs = model_embs
    self.trainable_embs = nn.Parameter(trainable_embs)

  def forward(self, x: Tensor) -> Tensor:
    """
    New embedding layer w/ added trainable soft prompt

    Args:
      x: token IDs to embed of shape `(batch_size, seq_len)`

    Returns:
      Tensor for embedded tokens w/ concat'd trainable soft prompt
    """

    input_embs = self.model_embs(x[:, self.trainable_embs.shape[1] :])
    return torch.cat(
      [
        repeat(
          self.trainable_embs,
          "b k d -> (repeat b) k d",
          repeat=input_embs.shape[0],
        ),
        input_embs,
      ],
      dim=1,
    )


# %%
class DocDataset(Dataset):
  def __init__(
    self,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    orig_prompt: str | Tensor,
    optim_prompt: str | Tensor,
    n_docs: int,
    doc_len: int,
    gen_batch_size: int = 10,
    validate_prompt: bool = True,
  ) -> None:
    if isinstance(orig_prompt, str):
      self.orig_wrapped_prompt, self.orig_prompt_slice = build_prompt(
        model.config.name_or_path, orig_prompt, tokenizer, validate_prompt
      )
    else:
      self.orig_wrapped_prompt = orig_prompt
      self.orig_prompt_slice = slice(0, orig_prompt.shape[-1])

    self.orig_wrapped_prompt = self.orig_wrapped_prompt.to(model.device)
    self.orig_doc_slice = slice(
      self.orig_wrapped_prompt.shape[-1],
      self.orig_wrapped_prompt.shape[-1] + doc_len,
    )

    if isinstance(optim_prompt, str):
      self.wrapped_prompt, self.prompt_slice = build_prompt(
        model.config.name_or_path, optim_prompt, tokenizer, validate_prompt
      )
    else:
      self.wrapped_prompt = optim_prompt
      self.prompt_slice = slice(0, optim_prompt.shape[-1])

    self.wrapped_prompt = self.wrapped_prompt.to(model.device)
    self.doc_slice = slice(
      self.wrapped_prompt.shape[-1],
      self.wrapped_prompt.shape[-1] + doc_len,
    )

    self.train_docs = self._gen_docs(model, n_docs, doc_len, gen_batch_size)
    self.dev_docs = self._gen_docs(model, n_docs, doc_len, gen_batch_size)

  def _gen_docs(
    self, model: PreTrainedModel, n_docs: int, doc_len: int, gen_batch_size: int
  ) -> Tensor:
    """
    Generate continuations (with just sampling, no constraints)

    Args:
      model: model to gen with
      n_docs: number of continuations to gen
      doc_len: length of each continuation
      gen_batch_size: batch size for gen

    Returns:
      doc tokens `(n_docs, doc_len)`
    """
    attn_mask = torch.ones_like(self.orig_wrapped_prompt).to(model.device)
    docs = torch.zeros((n_docs, doc_len), dtype=torch.long, device=model.device)

    for i in range(0, n_docs, gen_batch_size):
      cur_bsz = min(gen_batch_size, n_docs - i)

      # Generate docs w.r.t the original prompt
      doc_ids = model.generate(
        self.orig_wrapped_prompt,
        do_sample=True,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        min_new_tokens=doc_len,
        max_new_tokens=doc_len,
        num_return_sequences=cur_bsz,
        attention_mask=attn_mask,
      )

      # Without using HF generate method
      # doc_ids = torch.zeros((cur_bsz, doc_len), dtype=torch.long, device=model.device)
      # cur_prompt = (
      #   repeat(self.orig_wrapped_prompt, "b k -> (repeat b) k", repeat=cur_bsz)
      #   .clone()
      #   .to(model.device)
      # )
      # for j in range(doc_len):
      #   cur_logits = model(cur_prompt).logits
      #   cur_logits = cur_logits[:, -1, :]
      #   cur_logits[..., model.config.bos_token_id] = -float("inf")
      #   cur_logits[..., model.config.eos_token_id] = -float("inf")
      #   cur_probs = F.softmax(cur_logits, dim=-1)
      #   cur_tok = torch.multinomial(cur_probs, 1)
      #   doc_ids[:, j] = rearrange(cur_tok, "k b -> b k")
      # self.docs[i : i + cur_bsz] = doc_ids

      docs[i : i + cur_bsz] = doc_ids[:, self.orig_doc_slice]

    return docs

  def __len__(self) -> int:
    return self.train_docs.shape[0]

  def __getitem__(self, idx: int) -> dict:
    return {
      "optim_seq": torch.cat([self.wrapped_prompt[0], self.train_docs[idx]], dim=-1),
      "orig_seq": torch.cat(
        [self.orig_wrapped_prompt[0], self.train_docs[idx]], dim=-1
      ),
      "optim_seq_dev": torch.cat(
        [self.wrapped_prompt[0], self.dev_docs[idx]], dim=-1
      ),
      "orig_seq_dev": torch.cat(
        [self.orig_wrapped_prompt[0], self.dev_docs[idx]], dim=-1
      ),
    }


# %%
def compute_neg_log_prob(
  model: PreTrainedModel, seq: Tensor, pred_slice: slice, target_slice: slice
) -> Tensor:
  """
  Compute negative log prob of a slice in a sequence

  Args:
    model: the model
    seq: full input sequence `(batch_size, n_toks)`
    pred_slice: slice where logits are predicting
    target_slice: slice with labels for the predicting logits

  Returns:
    negative log probability `(n_toks,)`
  """

  pred_logits = model(seq).logits[:, pred_slice, :]
  log_probs = -F.cross_entropy(
    rearrange(pred_logits, "b k v -> b v k"), seq[:, target_slice], reduction="none"
  )

  return log_probs


# %%
def compute_grads(
  model: PreTrainedModel,
  seq: Tensor,
  prompt_slice: slice,
  doc_slice: slice,
  gamma: float = 0.0,
) -> Tensor:
  """
  Compute gradients for each token being optimized

  Args:
    model: the model
    seq: full sequence (prompt + docs) to compute grad
    prompt_slice: the unwrapped user prompt slice in the sequence
    doc_slice: the doc locations in the sequence
    gamma: fluency penalty gamma

  Returns:
    `(n_optim_toks, vocab_size)` the grads for each token
  """

  model_embs = model.get_input_embeddings().weight

  one_hot_suffix = torch.zeros(
    seq.shape[0],
    prompt_slice.stop - prompt_slice.start,
    model_embs.shape[0],
    device=model.device,
    dtype=model_embs.dtype,
  )
  one_hot_suffix.scatter_(-1, rearrange(seq[:, prompt_slice], "b k -> b k 1"), 1)
  one_hot_suffix.requires_grad = True

  suffix_embs = one_hot_suffix @ model_embs
  embs = model.get_input_embeddings()(seq).detach()
  full_embs = torch.cat(
    [
      embs[:, : prompt_slice.start, :],
      suffix_embs,
      embs[:, prompt_slice.stop :, :],
    ],
    dim=1,
  )

  logits = model(inputs_embeds=full_embs).logits
  targets = seq[:, doc_slice]
  loss_slice = slice(doc_slice.start - 1, doc_slice.stop - 1)

  prompt_pred_slice = slice(prompt_slice.start, prompt_slice.stop - 1)
  prompt_target_slice = slice(
    prompt_pred_slice.start + 1, prompt_pred_slice.stop + 1
  )
  fluency_penalty = (
    gamma
    * F.cross_entropy(
      rearrange(logits[0, prompt_pred_slice, :], "k v -> 1 v k"),
      rearrange(seq[0, prompt_target_slice], "k -> 1 k"),
      reduction="none",
    )
    .sum(dim=-1)
    .sum(dim=0)
  )  # sum over all tokens in prompt (only take first [0] since the log prob prompt is same for all docs ) NOTE: 2nd sum here does nothing, just reduces dim to be correct

  loss = F.cross_entropy(
    rearrange(logits[:, loss_slice, :], "b k v -> b v k"), targets
  )
  loss += fluency_penalty
  loss.backward()

  # mean across docs dim
  return one_hot_suffix.grad.clone().mean(dim=0)  # type: ignore


# %%
def replace_tok(
  model: PreTrainedModel,
  dataset: DocDataset,
  batch_size: int = 10,
  k: int = 256,
  gamma: float = 0.0,
) -> tuple[Tensor, float, float]:
  """
  Perform exact loss computations and token replacement

  Args:
    model: the model
    dataset: the dataset
    batch_size: batch size for doc forward pass
    k: top-k grads to keep
    gamma: fluency penalty gamma

  Returns:
    the new prompt IDs `(1, n_toks)` and the best lowest loss and the log prob. prompt
  """

  grads = torch.zeros(
    (
      dataset.prompt_slice.stop - dataset.prompt_slice.start,
      model.config.vocab_size,
    ),
    device=model.device,
  )

  for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
    seq = batch["optim_seq"]
    grads += compute_grads(
      model, seq, dataset.prompt_slice, dataset.doc_slice, gamma
    )

  grads /= grads.norm(dim=-1, keepdim=True)
  _, top_k_indices = torch.topk(-grads, k=k, dim=-1)

  with torch.no_grad():
    # 1 proposal for each position in the optim prompt
    n_proposals = top_k_indices.shape[0]
    grad_indices = rearrange(
      torch.randint(
        0, top_k_indices.shape[-1], (n_proposals,), device=top_k_indices.device
      ),
      "k -> 1 k",
    )
    positions = torch.arange(n_proposals, device=top_k_indices.device)
    new_toks = rearrange(top_k_indices[positions, grad_indices], "b k -> k b")
    positions = rearrange(positions, "k -> k 1")
    proposals = repeat(
      dataset.wrapped_prompt[:, dataset.prompt_slice],
      "b k -> (repeat b) k",
      repeat=n_proposals,
    ).clone()
    proposals = proposals.scatter_(-1, positions, new_toks)

    proposal_losses = torch.zeros((n_proposals,), device=model.device)
    # log_prob_prompts = torch.zeros((n_proposals,), device=model.device)
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
      seq = repeat(
        batch["optim_seq"], "b k -> (repeat b) k", repeat=proposals.shape[0]
      ).clone()
      # Now we have batch [replaced_tok_1 + doc_1, replaced_tok_2 + doc_2, ...]
      # for each n_proposals * batch_size
      seq[:, dataset.prompt_slice] = repeat(
        proposals,
        "b k -> (repeat b) k",
        repeat=seq.shape[0] // proposals.shape[0],
      )
      logits = model(seq).logits
      loss_slice = slice(dataset.doc_slice.start - 1, dataset.doc_slice.stop - 1)
      targets = seq[:, dataset.doc_slice]
      loss = F.cross_entropy(
        rearrange(logits[:, loss_slice, :], "b k v -> b v k"),
        targets,
        reduction="none",
      ).mean(dim=-1)
      # Split so that the docs are correctly added (split by num proposals, then sum across the docs)
      loss = rearrange(loss, "(k b) -> k b", b=n_proposals)
      loss = loss.sum(dim=0)
      proposal_losses += loss

    # avg across ALL docs
    proposal_losses /= len(dataset)

    # Factor in fluency penalty
    logits = model(proposals).logits
    nlls = F.cross_entropy(
      rearrange(logits[:, : proposals.shape[-1] - 1, :], "b k v -> b v k"),
      proposals[:, 1:],
      reduction="none",
    ).sum(dim=-1)  # sum over all tokens in prompt
    proposals_fluency = gamma * nlls
    proposal_losses += proposals_fluency

    best_idx = proposal_losses.argmin(dim=0)
    best_proposal = proposals[best_idx]
    best_loss = proposal_losses.min(dim=0).values
    return (
      rearrange(best_proposal, "k -> 1 k"),
      best_loss.item(),
      nlls[best_idx].item(),
    )


# %%
class OrigModelEmbs:
  """
  Switch to original model embeddings instead of the soft prompt layer
  """

  def __init__(
    self, model: PreTrainedModel, orig_embs: nn.Module, new_embs: nn.Module
  ):
    self.model = model
    self.orig_embs = orig_embs
    self.new_embs = new_embs

  def __enter__(self):
    self.model.set_input_embeddings(self.orig_embs)

  def __exit__(self, exception_type, exception_value, exception_traceback):
    self.model.set_input_embeddings(self.new_embs)


# %%
@torch.no_grad()
def compute_dataset_kl(
  model: PreTrainedModel,
  dataset: DocDataset,
  batch_size: int,
  embs: Tensor | None = None,
) -> tuple[float, float]:
  """
  Compute KL b/w orig and optim prompt using the holdout docs

  Args:
    dataset: the dataset
    batch_size: batch size for doc forward pass
    embs: if using soft prompts, compute from embs

  Returns:
    kl and std. dev.
  """
  doc_kls = Tensor([]).to(model.device)

  for batch in DataLoader(dataset, batch_size=batch_size):
    orig_pred_slice = slice(
      dataset.orig_doc_slice.start - 1, dataset.orig_doc_slice.stop - 1
    )
    pred_slice = slice(dataset.doc_slice.start - 1, dataset.doc_slice.stop - 1)
    orig_target_slice = dataset.orig_doc_slice
    target_slice = dataset.doc_slice

    neg_log_p_orig = compute_neg_log_prob(
      model, batch["orig_seq_dev"], orig_pred_slice, orig_target_slice
    ).sum(dim=-1)

    if embs is None:
      neg_log_p = compute_neg_log_prob(
        model, batch["optim_seq_dev"], pred_slice, target_slice
      ).sum(dim=-1)
    else:
      prefix_embs = model.get_input_embeddings()(
        batch["optim_seq_dev"][:, : dataset.prompt_slice.start]
      )
      suffix_embs = model.get_input_embeddings()(
        batch["optim_seq_dev"][:, dataset.prompt_slice.stop :]
      )
      full_embs = torch.cat(
        [
          prefix_embs,
          repeat(
            embs,
            "b k d -> (repeat b) k d",
            repeat=prefix_embs.shape[0],
          ),
          suffix_embs,
        ],
        dim=1,
      )

      logits = model(inputs_embeds=full_embs).logits
      neg_log_p = -F.cross_entropy(
        rearrange(logits[:, pred_slice, :], "b k v -> b v k"),
        batch["optim_seq_dev"][:, target_slice],
        reduction="none",
      ).sum(dim=-1)

    cur_kl = neg_log_p_orig - neg_log_p
    doc_kls = torch.cat([doc_kls, cur_kl])

  kl = doc_kls.mean().item()
  if kl < 0:
    print(f"WARNING: KL < 0: {kl}")

  std = doc_kls.std().item() / (doc_kls.shape[0] ** 0.5)

  return kl, std


# %%
def optim_gcg(
  model: PreTrainedModel,
  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
  dataset: DocDataset,
  n_epochs: int,
  kl_every: int,
  log_fpath: str,
  id_save_fpath: str,
  batch_size: int = 10,
  top_k: int = 256,
  gamma: float = 0.0,
  early_stop_kl: float = 0.0,
  suffix_mode: bool = False,
) -> tuple[list[dict], Tensor]:
  """
  Optimize a hard prompt via GCG

  Args:
    model: the model
    tokenizer: tokenizer
    dataset: the document/prompt dataset
    n_epochs: num epochs (number of token flips)
    kl_every: how often to compute KL
    log_fpath: file for logging progress
    id_save_fpath: file to save IDs
    batch_size: batch size for docs forward pass
    top_k: top-k for keeping in gradients
    gamma: natural prompt (fluency) penalty
    early_stop_kl: if KL goes below this threshold, stop optimization
    suffix_mode: if True, optimize a single document for a suffix

  Returns:
    list of progress log/results, and best optimized IDs `(1, n_optim_toks)`
  """

  print(
    f"\n\nTRAINING GCG:\n------------------------\nmodel: {model.config.name_or_path}\nnum epochs: {n_epochs}\nkl every: {kl_every}\ngamma: {gamma}\nearly stopping KL: {early_stop_kl}\n------------------------\n\n"
  )

  if suffix_mode:
    assert (
      dataset.train_docs.shape[0] == 1
    ), "Suffix mode should only have 1 doc: the suffix to optimize for"

  model.eval()
  pbar = tqdm(range(1, n_epochs + 1))
  to_ret = []
  best_loss = float("inf")
  if not suffix_mode:
    best_kl, best_std = compute_dataset_kl(model, dataset, batch_size=10)
  best_ids = dataset.wrapped_prompt[:, dataset.prompt_slice]
  cur_kl = None
  cur_std = None

  for i in pbar:
    ids, loss, log_prob_prompt = replace_tok(
      model, dataset=dataset, batch_size=batch_size, k=top_k, gamma=gamma
    )
    dataset.wrapped_prompt[:, dataset.prompt_slice] = ids

    if suffix_mode and loss < best_loss:
      best_ids = ids
      best_loss = loss
      torch.save(best_ids, id_save_fpath)
    elif not suffix_mode and i % kl_every == 0:
      cur_kl, cur_std = compute_dataset_kl(model, dataset, batch_size=batch_size)
      if cur_kl < best_kl:
        best_ids = ids
        best_kl = cur_kl
        best_std = cur_std
        torch.save(best_ids, id_save_fpath)

    best_loss = min(loss, best_loss)

    to_ret.append(
      {
        "epoch": i,
        "loss": loss,
        "best_loss": best_loss,
        "best_kl": best_kl,
        "best_std": best_std,
        "cur_kl": cur_kl if cur_kl is not None else best_kl,
        "cur_std": cur_std if cur_std is not None else best_std,
        "orig_prompt": tokenizer.decode(
          dataset.orig_wrapped_prompt[0, dataset.orig_prompt_slice]
        ),
        "prompt": tokenizer.decode(ids[0]),
        "nll_prompt": -log_prob_prompt,
      }
    )
    pbar.set_description(
      f"Epoch: {i}; Loss: {loss:.4f}; Best KL: {best_kl:.4f}; Cur KL: {(cur_kl if cur_kl is not None else best_kl):.4f}; NLL Prompt: {-log_prob_prompt:.4f}"
    )
    with open(log_fpath, "w") as f:
      json.dump(to_ret, f, indent=4, ensure_ascii=False)

    if best_kl < early_stop_kl:
      print(f"Early KL stopping <{early_stop_kl}")
      return to_ret, best_ids

  return to_ret, best_ids


# %%
def optim_soft(
  model: PreTrainedModel,
  dataset: DocDataset,
  n_epochs: int,
  kl_every: int,
  learning_rate: float,
  log_fpath: str,
  emb_save_fpath: str,
  batch_size: int = 10,
) -> tuple[list[dict], Tensor]:
  """
  Optimize a soft prompt

  Args:
    model: the model
    dataset: dataset to optim
    n_epochs: number of optim steps
    kl_every: how often to run KL validation
    learning_rate: lr
    log_fpath: file to log to
    emb_save_fpath: file to save embeddings to
    batch_size: size for doc pass
  """

  print(
    f"\n\nTRAINING SOFT:\n------------------------\nmodel: {model.config.name_or_path}\nnum epochs: {n_epochs}\nkl every: {kl_every}\n------------------------\n\n"
  )

  model.eval()
  to_ret = []
  pbar = tqdm(range(1, n_epochs + 1))
  best_embs = model.get_input_embeddings()(
    dataset.wrapped_prompt[:, dataset.prompt_slice]
  ).detach()
  orig_embs = model.get_input_embeddings()
  new_model_embs = SoftPromptEmbeddingLayer(model.get_input_embeddings(), best_embs)
  model.set_input_embeddings(new_model_embs)
  optimizer = torch.optim.Adam(
    [new_model_embs.trainable_embs],
    lr=learning_rate,
    eps=1e-4 if model.dtype != torch.float32 else 1e-8,
  )

  with OrigModelEmbs(model, orig_embs, new_model_embs):
    best_kl, best_std = compute_dataset_kl(
      model,
      dataset,
      batch_size=batch_size,
      embs=new_model_embs.trainable_embs.detach().clone(),
    )
  cur_kl = None
  cur_std = None

  for i in pbar:
    epoch_loss = 0.0

    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
      with OrigModelEmbs(model, orig_embs, new_model_embs):
        prefix_embs = model.get_input_embeddings()(
          batch["optim_seq"][:, : dataset.prompt_slice.start]
        )
        suffix_embs = model.get_input_embeddings()(
          batch["optim_seq"][:, dataset.prompt_slice.stop :]
        )

      embs = repeat(
        new_model_embs.trainable_embs,
        "b k d -> (repeat b) k d",
        repeat=prefix_embs.shape[0],
      )
      full_embs = torch.cat(
        [
          prefix_embs,
          embs,
          suffix_embs,
        ],
        dim=1,
      )
      logits = model(inputs_embeds=full_embs).logits
      pred_slice = slice(dataset.doc_slice.start - 1, dataset.doc_slice.stop - 1)
      target_slice = dataset.doc_slice

      loss = F.cross_entropy(
        rearrange(logits[:, pred_slice, :], "b k v -> b v k"),
        batch["optim_seq"][:, target_slice],
        reduction="none",
      ).sum()
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      epoch_loss += loss.item()

    if i % kl_every == 0:
      with OrigModelEmbs(model, orig_embs, new_model_embs):
        cur_kl, cur_std = compute_dataset_kl(
          model,
          dataset,
          batch_size=batch_size,
          embs=new_model_embs.trainable_embs.detach().clone(),
        )
      if cur_kl < best_kl:
        best_embs = new_model_embs.trainable_embs.detach().clone()
        best_kl = cur_kl
        best_std = cur_std
        torch.save(best_embs, emb_save_fpath)

    to_ret.append(
      {
        "epoch": i,
        "loss": epoch_loss,
        "best_kl": best_kl,
        "best_std": best_std,
        "cur_kl": cur_kl if cur_kl is not None else best_kl,
        "cur_std": cur_std if cur_std is not None else best_std,
      }
    )
    with open(log_fpath, "w") as f:
      json.dump(to_ret, f, indent=4)
    pbar.set_description(
      f"Epoch: {i}; Loss: {epoch_loss:.4f}; Best KL: {best_kl:.4f}; Cur KL: {cur_kl:.4f}"
    )

  return to_ret, best_embs
