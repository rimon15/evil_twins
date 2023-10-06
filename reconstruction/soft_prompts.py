from reconstruction.reconstruct import Reconstructor
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import join
from pathlib import Path
from einops import rearrange
import json
from reconstruction.common import (
    PROMPT_TEMPLATES,
    MODEL_NAME_OR_PATH_TO_NAME,
)
from typing import Optional
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import reconstruction.common as common
import pickle
import numpy as np
import random


class CorpusDataset(Dataset):
    def __init__(
        self,
        prompt: str,
        docs: list[str] | torch.Tensor,
        model_name_or_path: str,
        suffix: Optional[str] = None,
        prompt_ident: Optional[int] = None,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model_name = MODEL_NAME_OR_PATH_TO_NAME[model_name_or_path]

        if not suffix:
            self.prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            self.suffix_slice = None
        else:
            self.prompt_ids, self.suffix_slice = common.build_prompt(
                model_name, suffix, tokenizer
            )

        if isinstance(docs, list):
            docs_toks = tokenizer(
                docs,
                return_tensors="pt",
                add_special_tokens=False,
                max_length=32,
                truncation=True,
            )
            self.docs_ids = docs_toks["input_ids"]
            self.docs_attn_mask = docs_toks["attention_mask"]
        else:
            self.docs_ids = docs
            self.docs_attn_mask = torch.ones(docs.shape)
        self.prompt_ident = prompt_ident

    def __len__(self):
        return self.docs_ids.shape[0]

    def __getitem__(self, idx):
        out = {}

        out["prompt_ids"] = self.prompt_ids
        if self.suffix_slice is not None:
            out["suffix_slice"] = torch.tensor(
                [self.suffix_slice.start, self.suffix_slice.stop]
            )
        out["docs_ids"] = self.docs_ids[idx]
        out["docs_attn_mask"] = self.docs_attn_mask[idx]
        return out


class SoftReconstructor(Reconstructor):
    """
    Soft prompt reconstruction
    """

    def __init__(
        self,
        lr: float,
        num_epochs: int,
        early_stopping: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping

    def train(
        self,
        train_dataset: CorpusDataset,
        dev_dataset: CorpusDataset,
        suffix_only: bool,
        save_path: str,
        prompt_id: int,
        trial: int = 0,
    ) -> dict:
        """
        Optimization for soft prompt reconstruction.

        Params
        -----------
        train_dataset: CorpusDataset
            Dataset to train on, including control prompt and docs
        dev_dataset: CorpusDataset
            Dataset to evaluate on, including prompt and docs
        suffix_only: bool
            Whether to only optimize for the suffix slice in the prompt (otherwise randomly init all prompt embs and optimize that)
        save_path: str
            Path to save best prompt embeddings
        prompt_id: int
            ID of prompt
        trial: int
            Trial number
        """

        if suffix_only and train_dataset.suffix_slice is None:
            raise ValueError(
                "Suffix only reconstruction requires a suffix slice in the input prompt"
            )

        to_ret = []
        self.model.eval()  # Turn off dropout
        for param in self.model.parameters():
            param.requires_grad = False

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
        best_loss = float("inf")
        best_kl = float("inf")

        orig_prompt_embs = self.model.get_input_embeddings()(
            dev_dataset.prompt_ids.to(self.model.device)
        )

        if suffix_only:
            template_prefix_ids = train_dataset.prompt_ids[
                :, : train_dataset.suffix_slice.start
            ].to(self.model.device)
            template_suffix_ids = train_dataset.prompt_ids[
                :, train_dataset.suffix_slice.stop :
            ].to(self.model.device)
            optim_suffix_ids = train_dataset.prompt_ids[
                :, train_dataset.suffix_slice
            ].to(self.model.device)

            template_prefix_embs = self.model.get_input_embeddings()(
                template_prefix_ids
            )
            template_suffix_embs = self.model.get_input_embeddings()(
                template_suffix_ids
            )

            embs_to_optim = self.model.get_input_embeddings()(optim_suffix_ids)
            embs_to_optim.requires_grad = True
            optimizer = torch.optim.Adam([embs_to_optim], lr=self.lr, eps=1e-4)
            prompt_embs = torch.cat(
                [template_prefix_embs, embs_to_optim, template_suffix_embs],
                dim=1,
            )

        else:
            vocab = torch.ones(self.model.config.vocab_size)
            if self.remove_bos:
                vocab[self.model.config.bos_token_id] = 0
            if self.remove_eos:
                vocab[self.model.config.eos_token_id] = 0

            prompt_ids = torch.multinomial(vocab, orig_prompt_embs.shape[1])
            prompt_embs = self.model.get_input_embeddings()(
                prompt_ids.to(self.model.device)
            )
            prompt_embs = rearrange(prompt_embs, "k d -> 1 k d")
            prompt_embs.requires_grad = True
            optimizer = torch.optim.Adam([prompt_embs], lr=self.lr, eps=1e-4)

        curr_kl, curr_std_dev = self.compute_kl(
            prompt1=orig_prompt_embs,
            prompt2=prompt_embs,
            docs=dev_dataset.docs_ids,
            docs_attn_mask=None,
            p1_attn_mask=None,
            p2_attn_mask=None,
        )
        best_kl = curr_kl

        # Want to maximize sum of log probability of documents given prompt embedding
        pbar = tqdm(range(self.num_epochs), total=self.num_epochs)
        epoch_losses = []
        kl_losses = [(0, curr_kl, curr_std_dev)]
        to_ret.append(
            {
                "epoch": 0,
                "loss": 0,
                "kl": curr_kl,
                "std_dev": curr_std_dev,
            }
        )
        for epoch in pbar:
            epoch_loss = 0.0

            for batch in train_dataloader:
                if suffix_only:  # Update the new embeddings for the suffix each time
                    prompt_embs = torch.cat(
                        [template_prefix_embs, embs_to_optim, template_suffix_embs],
                        dim=1,
                    )

                loss = -self.log_prob_docs(
                    prompt_ids_or_embs=prompt_embs,
                    docs=batch["docs_ids"],
                    prompt_attn_mask=None,
                    docs_attn_mask=None,
                )
                loss = loss.sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

            epoch_losses.append(epoch_loss)
            pbar.set_description(
                f"Epoch loss: {epoch_loss}; Curr KL = {curr_kl} +- {curr_std_dev}"
            )

            if (epoch + 1) % self.kl_every == 0:
                curr_kl, curr_std_dev = self.compute_kl(
                    prompt1=orig_prompt_embs,
                    prompt2=prompt_embs,
                    docs=dev_dataset.docs_ids,
                    docs_attn_mask=None,
                    p1_attn_mask=None,
                    p2_attn_mask=None,
                )
                to_ret.append(
                    {
                        "epoch": epoch + 1,
                        "loss": epoch_loss,
                        "kl": curr_kl,
                        "std_dev": curr_std_dev,
                    }
                )
                kl_losses.append((epoch, curr_kl, curr_std_dev))

                if curr_kl < best_kl:
                    best_kl = curr_kl
                    torch.save(
                        prompt_embs,
                        join(
                            save_path,
                            f"soft_embs_prompt_id_{prompt_id}_len_{train_dataset.docs_ids.shape[-1]}_docs_{train_dataset.docs_ids.shape[0]}_trial_{trial}.pt",
                        ),
                    )

                if epoch_loss < best_loss:
                    best_loss = epoch_loss

                pbar.set_description(
                    f"Epoch loss: {epoch_loss}; Curr KL = {curr_kl} +- {curr_std_dev}"
                )

                if self.early_stopping and curr_kl > best_kl:
                    print("Early stopping after epoch ", epoch)
                    break

        return {
            "prompt_id": prompt_id,
            "trial": trial,
            "results": to_ret,
        }

    def load_datasets(
        self,
        dataset: str | list,
        suffix_only: bool,
        load_pkl: bool = True,
    ) -> None:
        """
        Load datasets to reconstructor
        """

        assert suffix_only, "Reconstruction now always requires suffix"

        if isinstance(dataset, str):
            if load_pkl:
                with open(dataset, "rb") as f:
                    dataset_lst = pickle.load(f)
            else:
                raise NotImplementedError("JSON loading is no longer supported")
                with open(dataset, "r") as f:
                    dataset = json.load(f)
        else:
            dataset_lst = dataset

        data = []
        for d in dataset_lst:
            if suffix_only:
                orig_prompt, suffix = common.gen_suffix_from_template(
                    self.model.config.name_or_path,
                    d["prompt"],
                    self.init_prompt_char,
                    self.optim_suffix_len,
                )

                if load_pkl:
                    d["train_docs"] = d["train_docs_tensor"]
                    d["dev_docs"] = d["dev_docs_tensor"]

                data.append(
                    (
                        CorpusDataset(
                            orig_prompt,
                            d["train_docs"],
                            self.model.config.name_or_path,
                            suffix,
                            d["id"],
                        ),
                        CorpusDataset(
                            d["prompt"],
                            d["dev_docs"],
                            self.model.config.name_or_path,
                            None,
                            d["id"],
                        ),
                    )
                )

            else:
                data.append(
                    (
                        CorpusDataset(
                            d["prompt"],
                            d["train_docs"],
                            self.model.config.name_or_path,
                            None,
                            d["id"],
                        ),
                        CorpusDataset(
                            d["prompt"],
                            d["dev_docs"],
                            self.model.config.name_or_path,
                            None,
                            d["id"],
                        ),
                    )
                )

        self.datasets = data
