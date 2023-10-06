from reconstruction.soft_prompts import SoftReconstructor
import reconstruction.common as common

from argparse import ArgumentParser
from pathlib import Path
import json
import pickle
import os


def reconstructor_worker(
    rec: SoftReconstructor,
    save_dir: str,
    trial_num: int,
) -> list:
    """
    Soft reconstruction process

    Parameters
    ----------
        rec: SoftReconstructor
            Reconstructor object
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
    for dset in rec.datasets:
        results.append(
            rec.train(
                dset[0],
                dset[1],
                True,
                save_dir,
                dset[0].prompt_ident,
                trial_num,
            )
        )
        # common.free_cuda_memory()

    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--kl_every", type=int, default=5)
    parser.add_argument("--optim_suffix_len", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=10)

    args = parser.parse_args()
    pool = common.setup_multiproc_env()
    n_procs = pool._processes
    models, tokenizers = common.load_models_tokenizers_parallel(
        args.model_name_or_path, True
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
            "init_prompt_char": "!",
            "optim_suffix_len": args.optim_suffix_len,
            "kl_every": args.kl_every,
        }
        reconstructor = SoftReconstructor(
            args.learning_rate,
            args.num_epochs,
            args.early_stopping,
            **reconstruct_args,
        )
        reconstructor.load_datasets(dataset_per_proc[i], True, True)
        reconstructors.append(reconstructor)

    print("Running soft prompt experiments")
    print(f"Number of prompts: {len(dataset)}")
    print(f"Number of trials: {args.n_trials}")

    results: list = []
    for i in range(args.n_trials):
        for j in range(n_procs):
            results.append(
                pool.apply_async(
                    reconstructor_worker,
                    (
                        reconstructors[j],
                        args.output_dir,
                        i,
                    ),
                )
            )

    pool.close()
    pool.join()
    results = [r.get() for r in results]
    results_flat = [x for sublist in results for x in sublist]

    with open(os.path.join(args.output_dir, "soft_results.json"), "w") as f:
        json.dump(results_flat, f, indent=4)
