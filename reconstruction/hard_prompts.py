# GCG adapted from https://github.com/llm-attacks/llm-attacks

from reconstruction.reconstruct import Reconstructor
import torch
from dataclasses import dataclass
from tqdm import tqdm
import json
import os
from os.path import join
import reconstruction.common as common
import urllib3
import spacy
import pickle
import random


IGNORE_INDEX = -100


@dataclass
class FullPrompt:
    """
    Holds a single user prompt, documents, suffix
    """

    prompt_ids: torch.Tensor
    suffix_slice: slice
    # The targets are the docs
    target_prefix_slice: slice
    target_prefix_ids: torch.Tensor
    prompt_ident: int  # For bookkeeping purposes

    def update_suffix(self, suffix_ids: torch.Tensor) -> None:
        """
        Updates the prompt with a new suffix
        """

        self.prompt_ids = torch.cat(
            [
                self.prompt_ids[:, : self.suffix_slice.start],
                suffix_ids.unsqueeze(0).to(self.prompt_ids.device),
                self.prompt_ids[:, self.suffix_slice.stop :],
            ],
            dim=-1,
        )


class HardReconstructorGCG(Reconstructor):
    def __init__(
        self,
        num_epochs: int,
        k: int,
        n_proposals: int,
        natural_prompt_penalty_gamma: int = 0,  # If 0, no natural prompt penalty
        clip_vocab: bool = False,
        warm_start_file: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_epochs = num_epochs
        self.k = k
        self.num_proposals = n_proposals
        self.natural_prompt_penalty_gamma = natural_prompt_penalty_gamma
        self.clip_vocab = clip_vocab
        self.warm_start_file = warm_start_file

        if self.clip_vocab:
            words = (
                urllib3.PoolManager()
                .request("GET", "https://www.mit.edu/~ecprice/wordlist.10000")
                .data.decode("utf-8")
            )
            words_list = words.split("\n")
            # nlp = spacy.load("en_core_web_sm")
            # words_list = list(set(nlp.vocab.strings))

            self.english_mask = self.get_english_only_mask(words_list)

    def get_english_only_mask(
        self,
        words_list: list[str],
    ) -> torch.Tensor:
        """
        Get english only tokens from the model's tokenizer
        """

        if "vocab_size" not in self.model.__dict__:
            vocab_size = self.model.config.vocab_size
        else:
            vocab_size = self.model.vocab_size

        english_only_mask = torch.zeros(vocab_size)
        for word in tqdm(
            words_list, desc="Building english only mask", total=len(words_list)
        ):
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            for word_id in word_ids:
                english_only_mask[word_id] = 1

        return english_only_mask

    def gcg_gradients(
        self,
        sample: FullPrompt,
    ) -> torch.Tensor:
        """
        First part of GCG. Compute gradients of each suffix (or all) tokens in the sample for Greedy Coordinate Gradient

        Parameters
        ----------
            sample: FullPrompt
                Prompt to compute gradients for

        Returns
        -------
            torch.Tensor
                Gradients of the suffix tokens (suffix_len, vocab_size)
        """

        assert (
            len(sample.prompt_ids.shape) == 2
        ), "prompt_ids must be of shape (1, seq_len)"

        orig_input_ids = sample.prompt_ids[0].to(self.model.device)
        n_docs = sample.target_prefix_ids.shape[0]
        if "vocab_size" not in self.model.__dict__:
            vocab_size = self.model.config.vocab_size
        else:
            vocab_size = self.model.vocab_size

        grads = torch.zeros(
            n_docs,
            sample.suffix_slice.stop - sample.suffix_slice.start,
            vocab_size,
            device=self.model.device,
        )

        for i in range(0, n_docs, self.batch_size):
            cur_batch_size = min(self.batch_size, n_docs - i)
            input_ids = orig_input_ids.repeat(cur_batch_size, 1)
            target_docs = sample.target_prefix_ids[i : i + cur_batch_size]

            input_ids = torch.cat([input_ids, target_docs.to(input_ids.device)], dim=1)

            model_embeddings = self.model.get_input_embeddings().weight
            one_hot_suffix = torch.zeros(
                input_ids.shape[0],
                sample.suffix_slice.stop - sample.suffix_slice.start,
                model_embeddings.shape[0],
                device=self.model.device,
                dtype=model_embeddings.dtype,
            )
            one_hot_suffix.scatter_(
                -1,
                input_ids[:, sample.suffix_slice].unsqueeze(-1),
                1,
            )
            one_hot_suffix.requires_grad = True
            suffix_embs = one_hot_suffix @ model_embeddings
            embeds = self.model.get_input_embeddings()(input_ids).detach()

            full_embs = torch.cat(
                [
                    embeds[:, : sample.suffix_slice.start, :],
                    suffix_embs,
                    embeds[:, sample.suffix_slice.stop :, :],
                ],
                dim=1,
            )

            logits = self.model(inputs_embeds=full_embs).logits
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            targets = input_ids[:, sample.target_prefix_slice]
            targets[targets == self.tokenizer.pad_token_id] = IGNORE_INDEX
            loss_slice = slice(
                sample.target_prefix_slice.start - 1,
                sample.target_prefix_slice.stop - 1,
            )

            loss = loss_fct(logits[:, loss_slice, :].transpose(1, 2), targets)

            # Add natural prompt penalty if specified
            loss += self.natural_prompt_penalty_gamma * -self.log_prob_prompt(
                orig_input_ids.unsqueeze(0), sample.suffix_slice
            )

            loss.backward()

            grads[i : i + cur_batch_size] = one_hot_suffix.grad.clone()

        return grads.mean(dim=0)

    @torch.no_grad()
    def proposal_loss(
        self,
        sample: FullPrompt,
        proposals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run forward pass with the new proposals and get the loss.

        Parameters
        ----------
            sample: FullPrompt
                Prompt to compute loss for
            proposals: torch.Tensor
                Proposals to compute loss for (num_proposals, prompt_len)

        Returns
        -------
            torch.Tensor
                Loss for each proposal (num_proposals,)
        """

        proposal_losses = torch.zeros(
            proposals.shape[0],
            device=proposals.device,
        )
        # Don't think we can batch across the proposal dimensions now since we have many documents (i.e. targets)
        for i in range(proposals.shape[0]):
            proposal = proposals[i].repeat(sample.target_prefix_ids.shape[0], 1)
            full_proposal_input = torch.cat(
                (
                    proposal.to(sample.target_prefix_ids.device),
                    sample.target_prefix_ids,
                ),
                dim=1,
            )
            proposal_embs = self.model.get_input_embeddings()(
                full_proposal_input.to(self.model.device)
            )
            logits = self.model(inputs_embeds=proposal_embs).logits
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=IGNORE_INDEX
            )
            loss_slice = slice(
                sample.target_prefix_slice.start - 1,
                sample.target_prefix_slice.stop - 1,
            )
            targets = full_proposal_input[:, sample.target_prefix_slice]
            targets[targets == self.tokenizer.pad_token_id] = IGNORE_INDEX
            loss = loss_fct(
                logits[:, loss_slice, :].transpose(1, 2), targets.to(logits.device)
            )
            proposal_losses[i] = loss.mean(dim=0).mean(
                dim=0
            )  # mean across all docs and then mean across all target tokens

        return proposal_losses

    def gcg_replace_tok(
        self,
        sample: FullPrompt,
    ) -> tuple[FullPrompt, float]:
        """
        This func implements part 2 of GCG. Now that we have the suffix tokens logits gradients w.r.t loss, we:
        For j = 0,...,total # of proposals
        1. Select the top-k logits for each suffix pos based on -grad of the logits
        2. For each proposal,
        3. Uniformly sample a random token in the top-k logits for replacement at position i
        4. Replace token i with the sampled token. Set this as proposal_j

        Run forward pass for all proposals, get the loss, and pick the proposal with the lowest loss.
        """

        # Compute gradients of the suffix tokens w.r.t the loss
        suffix_logits_grads = self.gcg_gradients(sample)
        suffix_logits_grads = suffix_logits_grads / suffix_logits_grads.norm(
            dim=-1, keepdim=True
        )

        if self.clip_vocab:
            # clip all non-english tokens
            suffix_logits_grads[:, self.english_mask != 1] = float("inf")

        # Select the top-k logits for each suffix pos based on -grad of the logits
        top_k_suffix_logits_grads, top_k_suffix_indices = torch.topk(
            -suffix_logits_grads,
            k=self.k,
            dim=-1,
        )

        self.num_proposals = min(self.num_proposals, top_k_suffix_indices.shape[0])
        proposals = sample.prompt_ids.repeat(self.num_proposals, 1).to(
            top_k_suffix_indices.device
        )

        rand_pos = torch.multinomial(
            torch.ones(
                suffix_logits_grads.shape[0],
                device=suffix_logits_grads.device,
            ),
            self.num_proposals,
            replacement=False,
        )
        for i in range(self.num_proposals):
            proposal = proposals[i]
            proposal_suffix_ids = proposal[sample.suffix_slice]
            proposal_suffix_ids[rand_pos[i]] = torch.gather(
                top_k_suffix_indices[i],
                0,
                torch.randint(
                    0,
                    top_k_suffix_indices.shape[-1],
                    (1,),
                    device=top_k_suffix_indices.device,
                ),
            )
            proposals[i] = proposal

        # Now compute the loss for each proposal, and pick the next candidate as the lowest one
        proposal_losses = self.proposal_loss(sample, proposals)
        best_proposal_idx = proposal_losses.argmin()
        best_proposal = proposals[best_proposal_idx]

        # Now update the sample with the new suffix
        new_suffix = best_proposal[sample.suffix_slice]
        sample.update_suffix(new_suffix)

        return sample, proposal_losses.min().item()

    def load_datasets(
        self,
        dataset: str | list,
        suffix_only: bool,
        load_doc_tensors: bool = True,
    ) -> None:
        """
        Load a dataset from a pickle file into the reconstructor

        Parameters
        ----------
            dataset: str | list
                Path to the dataset file or the dataset itself
            suffix_only: bool
                required
            load_doc_tensors: bool
                required
        """

        assert suffix_only, "Reconstruction now always requires suffix"

        if isinstance(dataset, str):
            if load_doc_tensors:
                with open(dataset, "rb") as f:
                    dataset_lst = pickle.load(f)
            else:
                with open(dataset, "r") as f:
                    dataset_lst = json.load(f)
        else:
            dataset_lst = dataset

        data = []
        for d in dataset_lst:
            prompt_ids = self.tokenizer.encode(d["prompt"], return_tensors="pt")
            suffix_slice = slice(0, prompt_ids.shape[-1])
            train_prompt_ids = torch.zeros_like(prompt_ids)
            train_prompt_ids[0, :] = self.init_prompt_tok

            if load_doc_tensors:
                train_docs = d["train_docs_tensor"]
                dev_docs = d["dev_docs_tensor"]
            else:
                raise NotImplementedError("JSON loading is no longer supported")
                train_docs = self.tokenizer(
                    d["train_docs"],
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                )["input_ids"]
                dev_docs = self.tokenizer(
                    d["dev_docs"],
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                )["input_ids"]
                train_prompt_ids = torch.zeros_like(prompt_ids)
                train_prompt_ids[0, :] = self.init_prompt_tok

            if suffix_only:
                if self.warm_start_file != "" and self.warm_start_file is not None:
                    for val in json.load(open(self.warm_start_file, "r")):
                        if val["id"] == d["id"]:
                            suffix = random.choice(val["responses"])
                else:
                    orig_prompt, suffix = common.gen_suffix_from_template(
                        self.model.config.name_or_path,
                        d["prompt"],
                        self.init_prompt_char,
                        self.optim_suffix_len,
                    )

                train_prompt_ids, train_suffix_slice = common.build_prompt(
                    self.model.config.name_or_path, suffix, self.tokenizer
                )

            target_prefix_slice = slice(
                train_prompt_ids.shape[-1],
                train_prompt_ids.shape[-1] + train_docs.shape[-1],
            )

            data.append(
                (
                    FullPrompt(
                        prompt_ids=train_prompt_ids,
                        suffix_slice=train_suffix_slice,
                        target_prefix_slice=target_prefix_slice,
                        target_prefix_ids=train_docs,
                        prompt_ident=d["id"],
                    ),
                    FullPrompt(
                        prompt_ids=prompt_ids,
                        suffix_slice=suffix_slice,
                        target_prefix_slice=target_prefix_slice,
                        target_prefix_ids=dev_docs,
                        prompt_ident=d["id"],
                    ),
                )
            )

        self.datasets = data

    def train(
        self,
        train_sample: FullPrompt,
        dev_sample: FullPrompt,
        save_path: str,
        prompt_id: int,
        trial: int,
    ) -> dict:
        """
        Optimization for hard prompt reconstruction using GCG

        Parameters
        ----------
            train_sample: FullPrompt
                Prompt to train on
            dev_sample: FullPrompt
                Prompt to evaluate on
            save_path: str
                Path to save results to
            prompt_id: int
                ID of the prompt
            trial: int
                Trial number

        Returns
        -------
            dict
                Dictionary of results w/ prompt id, trial, and the losses/kls
        """

        pbar = tqdm(range(self.num_epochs), total=self.num_epochs)
        best_loss = float("inf")
        best_kl = float("inf")
        to_ret = []

        kl, std_dev = self.compute_kl(
            prompt1=dev_sample.prompt_ids,
            prompt2=train_sample.prompt_ids,
            docs=dev_sample.target_prefix_ids,
            docs_attn_mask=None,
            p1_attn_mask=None,
            p2_attn_mask=None,
        )
        log_prob_prompt = self.log_prob_prompt(
            dev_sample.prompt_ids, train_sample.suffix_slice
        )

        if self.warm_start_file != "" and self.warm_start_file is not None:
            print(
                f"(id {train_sample.prompt_ident}) Original prompt: {self.tokenizer.decode(dev_sample.prompt_ids[0])}"
            )
            print(
                f"(id {train_sample.prompt_ident}) Initial prompt: {self.tokenizer.decode(train_sample.prompt_ids[0])}"
            )

        to_ret.append(
            {
                "epoch": 0,
                "loss": 0,
                "kl": kl,
                "std_dev": std_dev,
                "suffix": self.tokenizer.decode(train_sample.prompt_ids[0]),
                "log_prob_prompt": log_prob_prompt,
            }
        )

        for i in pbar:
            sample, loss = self.gcg_replace_tok(train_sample)

            if (i + 1) % self.kl_every == 0:
                kl, std_dev = self.compute_kl(
                    prompt1=dev_sample.prompt_ids,
                    prompt2=sample.prompt_ids,
                    docs=dev_sample.target_prefix_ids,
                    docs_attn_mask=None,
                    p1_attn_mask=None,
                    p2_attn_mask=None,
                )
                log_prob_prompt = self.log_prob_prompt(
                    sample.prompt_ids, sample.suffix_slice
                )
                to_ret.append(
                    {
                        "epoch": i + 1,
                        "loss": loss,
                        "kl": kl,
                        "std_dev": std_dev,
                        "suffix": self.tokenizer.decode(sample.prompt_ids[0]),
                        "log_prob_prompt": log_prob_prompt,
                    }
                )

            if loss < best_loss:
                best_loss = loss
            if kl < best_kl:
                best_kl = kl
                torch.save(
                    sample.prompt_ids,
                    join(
                        save_path,
                        f"hard_ids_prompt_{prompt_id}_len_{train_sample.target_prefix_ids.shape[-1]}_docs_{train_sample.target_prefix_ids.shape[0]}_trial_{trial}.pt",
                    ),
                )

            pbar.set_description(
                f"Epoch loss:{loss:.2f};Best KL={best_kl:.2f};Curr KL={kl:.2f}+-{std_dev:.2f};Logprob. prompt={log_prob_prompt:.2f}"
            )

        return {
            "prompt_id": prompt_id,
            "trial": trial,
            "results": to_ret,
        }
