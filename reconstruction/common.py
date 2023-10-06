from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
import torch
import gc
import multiprocessing as mp
import os
from typing import Optional


PROMPT_TEMPLATES = {
    "vicuna": {
        "prefix": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: ",
        "suffix": "\nASSISTANT:",
    },
    "Llama-2-7b-chat-hf": {
        "prefix": "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n",
        "suffix": " [/INST]",
    },
    "opt": {
        "prefix": "",
        "suffix": "",
    },
    "phi": {
        "prefix": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: ",
        "suffix": "\nASSISTANT:",
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
}

MODEL_NAME_OR_PATH_TO_NAME = {
    "lmsys/vicuna-7b-v1.3": "vicuna",
    "lmsys/vicuna-7b-v1.5": "vicuna",
    "vicuna": "vicuna",
    "facebook/opt-350m": "opt",
    "facebook/opt-1.3b": "opt",
    "microsoft/phi-1_5": "phi",
    "teknium/Puffin-Phi-v2": "phi",
    "OpenAssistant/oasst-sft-1-pythia-12b": "oasst",
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
}

DEVICE_MAPS = {
    "llama_for_causal_lm_2_gpus": {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 0,
        "model.layers.4": 0,
        "model.layers.5": 0,
        "model.layers.6": 0,
        "model.layers.7": 0,
        "model.layers.8": 0,
        "model.layers.9": 0,
        "model.layers.10": 0,
        "model.layers.11": 0,
        "model.layers.12": 0,
        "model.layers.13": 0,
        "model.layers.14": 0,
        "model.layers.15": 0,
        "model.layers.16": 1,
        "model.layers.17": 1,
        "model.layers.18": 1,
        "model.layers.19": 1,
        "model.layers.20": 1,
        "model.layers.21": 1,
        "model.layers.22": 1,
        "model.layers.23": 1,
        "model.layers.24": 1,
        "model.layers.25": 1,
        "model.layers.26": 1,
        "model.layers.27": 1,
        "model.layers.28": 1,
        "model.layers.29": 1,
        "model.layers.30": 1,
        "model.layers.31": 1,
        "model.norm": 1,
        "lm_head": 1,
    },
    "pythia_2_gpus": {
        "gpt_neox.embed_in": 0,
        "gpt_neox.emb_dropout": 0,
        "gpt_neox.layers.0": 0,
        "gpt_neox.layers.1": 0,
        "gpt_neox.layers.2": 0,
        "gpt_neox.layers.3": 0,
        "gpt_neox.layers.4": 0,
        "gpt_neox.layers.5": 0,
        "gpt_neox.layers.6": 0,
        "gpt_neox.layers.7": 0,
        "gpt_neox.layers.8": 0,
        "gpt_neox.layers.9": 0,
        "gpt_neox.layers.10": 0,
        "gpt_neox.layers.11": 0,
        "gpt_neox.layers.12": 0,
        "gpt_neox.layers.13": 0,
        "gpt_neox.layers.14": 0,
        "gpt_neox.layers.15": 1,
        "gpt_neox.layers.16": 1,
        "gpt_neox.layers.17": 1,
        "gpt_neox.layers.18": 1,
        "gpt_neox.layers.19": 1,
        "gpt_neox.layers.20": 1,
        "gpt_neox.layers.21": 1,
        "gpt_neox.layers.22": 1,
        "gpt_neox.layers.23": 1,
        "gpt_neox.layers.24": 1,
        "gpt_neox.layers.25": 1,
        "gpt_neox.layers.26": 1,
        "gpt_neox.layers.27": 1,
        "gpt_neox.layers.28": 1,
        "gpt_neox.layers.29": 1,
        "gpt_neox.layers.30": 1,
        "gpt_neox.layers.31": 1,
        "gpt_neox.final_layer_norm": 1,
        "embed_out": 1,
    },
}


def build_prompt(
    model_name: str, suffix: str, tokenizer: PreTrainedTokenizer
) -> tuple[torch.Tensor, slice]:
    """
    Given the actual "suffix" (prompt), add in the prefix/suffix for the given instruction tuned model

    Parameters
    ----------
        model_name: str
            Model name or path
        suffix: str
            The actual prompt to wrap around
        tokenizer: PreTrainedTokenizer
            Tokenizer for the model

    Returns
    -------
        tuple[torch.Tensor, slice]
            Tuple of the prompt ids and the slice of the actual prompt (suffix)
    """

    model_name = MODEL_NAME_OR_PATH_TO_NAME[model_name]
    cur_prompt = PROMPT_TEMPLATES[model_name]["prefix"]
    suffix_start_idx = max(len(tokenizer(cur_prompt)["input_ids"]) - 1, 0)
    cur_prompt += suffix
    suffix_end_idx = len(tokenizer(cur_prompt)["input_ids"])
    cur_prompt += PROMPT_TEMPLATES[model_name]["suffix"]

    prompt_ids = tokenizer(cur_prompt, return_tensors="pt")["input_ids"]
    suffix_slice = slice(suffix_start_idx, suffix_end_idx)
    return prompt_ids, suffix_slice


def gen_suffix_from_template(
    model_name: str, prompt: str, suffix_char: str, suffix_len: int
) -> tuple[str, str]:
    """
    Given a fully wrapped prompt, replace the actual prompt (suffix) with the control tokens

    Parameters
    ----------
        model_name: str
            Model name or path
        prompt: str
            Prompt to extract suffix from
        suffix_char: str
            Character to use for suffix
        suffix_len: int
            Length of suffix

    Returns
    -------
        tuple[str, str]
            Tuple of the original prompt and the new suffix
    """

    pt = PROMPT_TEMPLATES[MODEL_NAME_OR_PATH_TO_NAME[model_name]]

    orig_prompt = prompt.replace(pt["suffix"], "")
    orig_prompt = orig_prompt.replace(pt["prefix"], "")
    suffix = ((suffix_char + " ") * suffix_len)[:-1]

    return orig_prompt, suffix


def free_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()


def load_model_tokenizer(
    model_name_or_path: str,
    fp16: bool = True,
    device_map: str | dict = "auto",
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return model, tokenizer


def load_models_tokenizers_parallel(
    model_name_or_path: str,
    fp16: bool = True,
    split_model_gpus: Optional[list[tuple[int, int]]] = None,
) -> tuple[list[AutoModelForCausalLM], list[PreTrainedTokenizer]]:
    """
    Load multiple models for parallel processing **CURRENTLY ONLY SUPPORTS SHARDING ACROSS 2 GPUS**

    Args:
        model_name_or_path (str): Model name or path
        fp16 (bool, optional): Whether to use fp16. Defaults to True.
        split_model (bool, optional): Whether to split the model across multiple (only 2 supported for now) GPUs. Used for hard reconstruction
    """

    models = []
    tokenizers = []

    print("Loading models...")

    if split_model_gpus:
        if (
            "vicuna" in MODEL_NAME_OR_PATH_TO_NAME[model_name_or_path]
            or "llama" in MODEL_NAME_OR_PATH_TO_NAME[model_name_or_path]
        ):
            dmap = DEVICE_MAPS["llama_for_causal_lm_2_gpus"]
        elif "pythia" in MODEL_NAME_OR_PATH_TO_NAME[model_name_or_path]:
            dmap = DEVICE_MAPS["pythia_2_gpus"]

        for device0, device1 in split_model_gpus:
            cur_dmap = {k: device0 if v == 0 else device1 for k, v in dmap.items()}
            model, tokenizer = load_model_tokenizer(
                model_name_or_path, fp16, device_map=cur_dmap
            )
            models.append(model)
            tokenizers.append(tokenizer)

    else:
        for i in range(torch.cuda.device_count()):
            model, tokenizer = load_model_tokenizer(
                model_name_or_path, fp16, device_map=f"cuda:{i}"
            )
            models.append(model)
            tokenizers.append(tokenizer)

    return models, tokenizers


def setup_multiproc_env(split_models: bool = False):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    mp.set_start_method("spawn")
    n_procs = (
        torch.cuda.device_count() // 2 if split_models else torch.cuda.device_count()
    )
    pool = mp.Pool(processes=n_procs)

    return pool


def split_for_multiproc(data: list, n_procs: int) -> list[list]:
    """
    Splits a list into n_procs chunks for multi GPU processing
    """

    n_samples = len(data)
    chunk_size = n_samples // n_procs
    return [data[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]
