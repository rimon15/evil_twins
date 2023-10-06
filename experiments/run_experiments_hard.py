from reconstruction.hard_prompts import HardReconstructorGCG
import reconstruction.common as common
import torch

import os
from typing import Optional
from argparse import ArgumentParser
from pathlib import Path
import pickle
import json


def reconstructor_worker(
    rec: HardReconstructorGCG,
    model_name_or_path: str,
    model_split: Optional[tuple[int, int]],
    save_dir: str,
    trial_num: int,
) -> list:
    """
    Reconstruction process for hard prompts

    Parameters
    ----------
        rec: HardReconstructorGCG
            Reconstructor object
        model_name_or_path: str
            Model name or path
        model_split: Optional[tuple[int, int]]
            If sharding the model across multiple GPUs, the GPUs to shard across (defined from common.py)
        save_dir: str
            Directory to save results to
        trial_num: int
            Current trial number for labeling purposes

    Returns
    -------
        list
            List of results of losses/KLs from Reconstructor.train()
    """

    results = []

    if model_split is not None:
        rec.model = common.load_models_tokenizers_parallel(
            model_name_or_path, True, [model_split]
        )[0][0]

    for dset in rec.datasets:
        results.append(
            rec.train(
                dset[0],
                dset[1],
                save_dir,
                dset[0].prompt_ident,
                trial_num,
            )
        )

    if model_split is not None:
        # Now free whatever gpus we allocated to for the next run...
        del rec.model
        common.free_cuda_memory()

    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--n_proposals", type=int, default=100)
    parser.add_argument("--natural_prompt_penalty", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--kl_every", type=int, default=5)
    parser.add_argument(
        "--optim_suffix_len",
        type=int,
        default=20,
        help="length of prompt to optimize, if warm start is provided this isn't used",
    )
    parser.add_argument("--init_prompt_char", type=str, default="!")
    parser.add_argument("--clip_vocab", action="store_true")
    parser.add_argument(
        "--warm_start_file",
        type=str,
        help="JSON with the ChatGPT suggested prompts for warm start",
    )
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--sharded", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    pool = common.setup_multiproc_env(args.sharded)
    n_procs = pool._processes

    if args.sharded:
        sharding_list = [(i, i + 1) for i in range(0, torch.cuda.device_count(), 2)]
        models, tokenizers = common.load_models_tokenizers_parallel(
            args.model_name_or_path, True, sharding_list
        )
    else:
        models, tokenizers = common.load_models_tokenizers_parallel(
            args.model_name_or_path, args.fp16
        )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dataset = pickle.load(open(args.dataset_path, "rb"))
    dataset_per_proc = common.split_for_multiproc(dataset, n_procs)

    reconstructors = []
    for i in range(n_procs):
        reconstruct_args = {
            "model": models[i],
            "tokenizer": tokenizers[i],
            "batch_size": args.batch_size,
            "remove_eos": True,
            "remove_bos": True,
            "init_prompt_char": args.init_prompt_char,
            "optim_suffix_len": args.optim_suffix_len,
            "kl_every": args.kl_every,
        }
        reconstructor = HardReconstructorGCG(
            args.num_epochs,
            args.top_k,
            args.n_proposals,
            args.natural_prompt_penalty,
            args.clip_vocab,
            args.warm_start_file,
            **reconstruct_args,
        )
        reconstructor.load_datasets(dataset_per_proc[i], True, True)

        if args.sharded:
            del reconstruct_args["model"]
            del (
                reconstructor.model
            )  # We cannot pass the model to the worker when sharding across multiple GPUs
            # https://github.com/huggingface/accelerate/issues/1598
            # https://github.com/NVIDIA/apex/issues/415
        reconstructors.append(reconstructor)

    print("Running hard prompt experiments")
    print(f"Number of prompts: {len(dataset)}")
    print(f"Number of trials: {args.n_trials}")

    # This is the workaround i came up with for not being able to pickle the model if its sharded across GPUs
    # the workers will load the model each time they are called, then delete it after... so hacky lmao
    # but its fine, the bottleneck isn't the model loading anyway
    if args.sharded:
        del models
        common.free_cuda_memory()

    results: list = []
    for i in range(args.n_trials):
        for j in range(n_procs):
            results.append(
                pool.apply_async(
                    reconstructor_worker,
                    (
                        reconstructors[j],
                        args.model_name_or_path,
                        sharding_list[j] if args.sharded else None,
                        args.output_dir,
                        i,
                    ),
                )
            )

    pool.close()
    pool.join()
    results = [x.get() for x in results]
    results_flat = [x for sublist in results for x in sublist]

    print("Saving results...")
    with open(os.path.join(args.output_dir, "hard_results.json"), "w") as f:
        json.dump(results_flat, f, indent=4, ensure_ascii=False)
