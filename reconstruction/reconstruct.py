from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from typing import Optional
from tqdm import tqdm
import torch.nn.functional as F
import json
from pathlib import Path
from os.path import join
from reconstruction.common import (
    MODEL_NAME_OR_PATH_TO_NAME,
    PROMPT_TEMPLATES,
    free_cuda_memory,
)
import pickle


class Reconstructor(object):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int,
        remove_bos: bool = True,
        remove_eos: bool = True,
        init_prompt_char: str = "!",
        optim_suffix_len: int = 20,
        kl_every: int = 5,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = batch_size
        self.remove_bos = remove_bos
        self.remove_eos = remove_eos
        self.kl_every = kl_every

        assert (
            self.remove_bos and self.remove_eos
        ), "Reconstruction only supported w/ BOS & EOS removal"

        self.optim_suffix_len = optim_suffix_len
        self.init_prompt_char = init_prompt_char
        self.init_prompt_tok = self.tokenizer.encode(
            self.init_prompt_char, return_tensors="pt"
        )[0, -1].item()

        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    def causal_forward(
        self,
        prompt_ids_or_embs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> CausalLMOutputWithPast:
        """
        Forward pass for a decoder-only model
        DON'T FORGET TO USE torch.no_grad() when calling this if necessary!

        Parameters
        ----------
            prompt_ids_or_embs: torch.Tensor
                Either a tensor of prompt ids or a tensor of prompt embeddings (batch_size, seq_len) or (batch_size, seq_len, embedding_dim)
            attention_mask: Optional[torch.Tensor]
                deprecated

        Returns
        -------
            CausalLMOutputWithPast
                Output of the model
        """

        assert (
            len(prompt_ids_or_embs.shape) == 3 or len(prompt_ids_or_embs.shape) == 2
        ), "prompt_ids_or_embs must be of shape (batch_size, seq_len) or (batch_size, seq_len, embedding_dim)"

        prompt_ids_or_embs = prompt_ids_or_embs.to(self.model.device)

        if len(prompt_ids_or_embs.shape) == 3:
            assert (
                attention_mask is None
            ), "Cannot use an attention mask with direct embeddings"
            outputs = self.model(
                inputs_embeds=prompt_ids_or_embs,
                return_dict=True,
            )
        else:
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)
            outputs = self.model(
                input_ids=prompt_ids_or_embs,
                attention_mask=attention_mask,
                return_dict=True,
            )

        if self.remove_bos:
            outputs.logits[:, :, self.model.config.bos_token_id] = -float(
                "inf"
            )  # Remove all BOS
        if self.remove_eos:
            outputs.logits[:, :, self.model.config.eos_token_id] = -float(
                "inf"
            )  # Remove all EOS

        return outputs

    def log_prob_docs(
        self,
        prompt_ids_or_embs: torch.Tensor,
        docs: torch.Tensor,
        prompt_attn_mask: Optional[torch.Tensor],
        docs_attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute log P(doc | prompt) given a causal LM & document batch.
        returns a tensor of shape num_docs for the log prob. of each doc

        The prompt must be either a hard prompt (i.e., a sequence of token ids) or a soft prompt (i.e., sequence of embeddings).

        Parameters
        ----------
            prompt_ids_or_embs: torch.Tensor
                Either a tensor of prompt ids or a tensor of prompt embeddings (batch_size, seq_len) or (batch_size, seq_len, embedding_dim)
            docs: torch.Tensor
                Tensor of document ids (num_docs, max_len)
            prompt_attn_mask: Optional[torch.Tensor]
                deprecated
            docs_attn_mask: Optional[torch.Tensor]
                deprecated

        Returns
        -------
            torch.Tensor
                Tensor of shape num_docs for the log prob. of each doc
        """

        assert len(prompt_ids_or_embs.shape) == 2 or len(prompt_ids_or_embs.shape) == 3

        if len(prompt_ids_or_embs.shape) == 2:
            prompt_embs = self.model.get_input_embeddings()(
                prompt_ids_or_embs.to(self.model.device)
            )
        else:
            prompt_embs = prompt_ids_or_embs

        assert (
            len(docs.shape) == 2
        ), "docs should be tensor of shape (num_docs, max_len)"
        prompt_len = prompt_embs.shape[1]
        assert (
            prompt_len > 0
        ), "There must be at least one token in the prompt, even if it is just the BOS token"

        log_prob_docs = torch.zeros(
            docs.shape[0], dtype=torch.float32, device=self.model.device
        )

        for i in range(0, docs.shape[0], self.batch_size):
            cur_batch_size = min(self.batch_size, docs.shape[0] - i)
            cur_docs = docs[i : i + cur_batch_size].to(self.model.device)
            cur_docs_embs = self.model.get_input_embeddings()(cur_docs)
            cur_prompt_embs = prompt_embs.clone()
            cur_prompt_embs = cur_prompt_embs.repeat(cur_docs.shape[0], 1, 1)
            cur_prompt_docs_embs = torch.cat((cur_prompt_embs, cur_docs_embs), dim=1)

            cur_attn_masks = None
            if prompt_attn_mask is not None and docs_attn_mask is not None:
                cur_prompt_attn_mask = prompt_attn_mask[i : i + cur_batch_size]
                cur_prompt_attn_mask = cur_prompt_attn_mask.repeat(cur_docs.shape[0], 1)
                cur_docs_attn_mask = docs_attn_mask[i : i + cur_batch_size]
                cur_attn_masks = torch.cat(
                    (
                        cur_prompt_attn_mask,
                        cur_docs_attn_mask,
                    ),
                    dim=1,
                )

            logits = self.causal_forward(
                prompt_ids_or_embs=cur_prompt_docs_embs,
                attention_mask=cur_attn_masks,
            ).logits  # num_docs x tot_len x vocab_size
            doc_logits = logits[
                :, prompt_len - 1 : -1, :
            ]  # cur_batch_size x doc_len x vocab_size

            # # Need to remove the padding indices from the loss
            labels = cur_docs.clone()
            labels[cur_docs == self.model.config.pad_token_id] = -100

            output = -self.loss_fct(doc_logits.transpose(-1, -2), labels)
            log_prob_docs[i : i + cur_batch_size] = torch.sum(output, dim=-1)

        return log_prob_docs

    def log_prob_prompt(
        self,
        prompt_ids: torch.Tensor,
        suffix_slice: slice,
    ) -> float:
        """
        Compute log P(prompt) given a causal LM and the slice for where the actual prompt is in the template

        Parameters
        ----------
            prompt_ids: torch.Tensor
                Tensor of prompt ids (1, seq_len)
            suffix_slice: slice
                Slice of the prompt (suffix) in the template

        Returns
        -------
            float
                Log prob. of the prompt
        """

        assert len(prompt_ids.shape) == 2

        prompt_ids = prompt_ids.to(self.model.device)
        target_slice = slice(max(suffix_slice.start - 1, 0), suffix_slice.stop - 1)

        logits = self.causal_forward(
            prompt_ids_or_embs=prompt_ids, attention_mask=None
        ).logits

        output = -self.loss_fct(
            logits[:, target_slice, :].reshape(-1, logits.shape[-1]),
            prompt_ids[:, target_slice].reshape(-1),
        )
        return torch.sum(output, dim=-1).item()

    @torch.no_grad()
    def compute_kl(
        self,
        prompt1: torch.Tensor,
        prompt2: torch.Tensor,
        docs: torch.Tensor,
        docs_attn_mask: Optional[torch.Tensor],
        p1_attn_mask: Optional[
            torch.Tensor
        ],
        p2_attn_mask: Optional[torch.Tensor],
        return_kls: bool = False,
    ) -> tuple[float, float]:
        """
        Estimate KL divergence b/w p1 and p2 given all docs D_i drawn from prompt1
        dist_KL = 1/n * sum_{i=1}^n log(P(D_i | prompt1)) - log(P(D_i | prompt2))

        Parameters
        ----------
            prompt1: torch.Tensor
                Tensor of prompt ids (1, seq_len)
            prompt2: torch.Tensor
                Tensor of prompt ids (1, seq_len)
            docs: torch.Tensor
                Tensor of document ids (num_docs, max_len)
            docs_attn_mask: Optional[torch.Tensor]
                deprecated
            p1_attn_mask: Optional[torch.Tensor]
                deprecated
            p2_attn_mask: Optional[torch.Tensor]
                deprecated

        Returns
        -------
            tuple[float, float]
                Tuple of the KL divergence and the std. dev. across all docs
        """

        # Compute log P(D_i | prompt1)
        log_p1 = self.log_prob_docs(
            prompt_ids_or_embs=prompt1,
            docs=docs,
            prompt_attn_mask=p1_attn_mask,
            docs_attn_mask=docs_attn_mask,
        )

        # Compute log P(D_i | prompt2)
        log_p2 = self.log_prob_docs(
            prompt_ids_or_embs=prompt2,
            docs=docs,
            prompt_attn_mask=p2_attn_mask,
            docs_attn_mask=docs_attn_mask,
        )

        doc_kls = log_p1 - log_p2
        if return_kls:
            return doc_kls
        kl = doc_kls.mean().item()
        std_dev = doc_kls.std().item() / (doc_kls.shape[0] ** 0.5)
        return kl, std_dev

    @torch.no_grad()
    def gen_docs(
        self,
        prompt: str,
        max_len: int,
        n_docs: int,
        gen_from_embs: bool = False,
        prompt_embs: Optional[torch.Tensor] = None,
        show_progress: bool = False,
    ) -> tuple[torch.Tensor, list[str]]:
        """
        Generate documents from a prompt

        Parameters
        ----------
            prompt: str
                Prompt to generate documents from
            max_len: int
                Number of tokens to generate for each document
            n_docs: int
                Number of documents to generate
            gen_from_embs: bool
                Whether to generate from embeddings or ids
            prompt_embs: Optional[torch.Tensor]
                If generating from embeddings, the embeddings to use
            show_progress: bool
                Whether to show progress bar

        Returns
        -------
            tuple[torch.Tensor, list[str]]
                Tuple of the generated documents (tensor of shape (n_docs, max_len)) and the generated documents (list of length n_docs)
        """

        self.model.eval()
        if gen_from_embs:
            assert (
                len(prompt_embs.shape) == 3
            ), "prompt_embs must be of shape (1, seq_len, embedding_dim)"

            prompt_embs = prompt_embs.to(self.model.device)
        else:
            prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            prompt_embs = self.model.get_input_embeddings()(
                prompt_ids.to(self.model.device)
            )

        gen_ids = torch.zeros(
            (n_docs, max_len), dtype=torch.long, device=self.model.device
        )

        for i in tqdm(
            range(0, n_docs, self.batch_size),
            desc="Generating docs",
            disable=not show_progress,
        ):
            cur_batch_size = min(self.batch_size, n_docs - i)
            cur_prompt_embs = prompt_embs.repeat(
                cur_batch_size, 1, 1
            )  # cur_batch_size x prompt_len x embedding_dim

            for j in range(max_len):
                logits = self.causal_forward(
                    prompt_ids_or_embs=cur_prompt_embs, attention_mask=None
                ).logits
                logits = logits[:, -1, :]

                next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                gen_ids[i : i + cur_batch_size, j] = next_tok[:, 0]

                next_tok_embs = self.model.get_input_embeddings()(next_tok)
                cur_prompt_embs = torch.cat((cur_prompt_embs, next_tok_embs), dim=1)

        return gen_ids.cpu(), self.tokenizer.batch_decode(gen_ids)

    def gen_datasets_from_prompts(
        self,
        prompts: list[tuple[int, str]],  # (id, prompt)
        max_len: int,
        n_docs: int,
        out_dir: str,
        dataset_name: str,
        show_progress: bool = False,
        gen_dev: bool = True,
        show_doc_progress: bool = False,
        save_tensors: bool = True,
        save_to_file: bool = False,
    ) -> list[dict]:
        """
        Generate documents from a list of prompts

        Parameters
        ----------
            prompts: list[tuple[int, str]]
                List of prompts to generate documents from
            max_len: int
                Number of tokens to generate for each document
            n_docs: int
                Number of documents to generate
            out_dir: str
                Directory to save datasets to
            dataset_name: str
                Name of the dataset
            show_progress: bool
                Whether to show progress bar
            gen_dev: bool
                Whether to generate dev docs
            show_doc_progress: bool
                Whether to show progress bar for generating docs
            save_tensors: bool
                Whether to save tensors
            save_to_file: bool
                Whether to save to file

        Returns
        -------
            list[dict]
                List of generated datasets
        """

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        dataset = []

        if not save_tensors:
            raise NotImplementedError("Not saving tensors is no longer supported")

        for id, prompt in tqdm(
            prompts,
            desc="Generating doc datasets from prompts",
            disable=not show_progress,
        ):
            name = MODEL_NAME_OR_PATH_TO_NAME[self.model.config.name_or_path]
            prompt = (
                PROMPT_TEMPLATES[name]["prefix"]
                + prompt
                + PROMPT_TEMPLATES[name]["suffix"]
            )
            train_docs_tensor, train_docs_str = self.gen_docs(
                prompt, max_len, n_docs, False, None, show_doc_progress
            )

            dev_docs_tensor = None
            dev_docs_str: list = []
            if gen_dev:
                dev_docs_tensor, dev_docs_str = self.gen_docs(
                    prompt, max_len, n_docs, False, None, show_doc_progress
                )

            dataset.append(
                {
                    "id": id,
                    "prompt": prompt,
                    "train_docs_str": train_docs_str,
                    "dev_docs_str": dev_docs_str,
                    "train_docs_tensor": train_docs_tensor,
                    "dev_docs_tensor": dev_docs_tensor,
                }
            )

        if save_to_file:
            print(
                "WARNING: saving to file is no longer recommended, use the returned dataset and save yourself"
            )
            with open(
                join(
                    out_dir,
                    f"{dataset_name}_prompts_{len(prompts)}_len_{max_len}_docs_{n_docs}.json",
                ),
                "w",
            ) as f:
                str_dataset = []
                for d in dataset:
                    str_dataset.append(
                        {
                            "id": d["id"],
                            "prompt": d["prompt"],
                            "train_docs_str": d["train_docs_str"],
                            "dev_docs_str": d["dev_docs_str"],
                        }
                    )
                json.dump(str_dataset, f, indent=4)

            if save_tensors:
                pickle.dump(
                    dataset,
                    open(
                        join(
                            out_dir,
                            f"{dataset_name}_prompts_{len(prompts)}_len_{max_len}_docs_{n_docs}.pkl",
                        ),
                        "wb",
                    ),
                )

        return dataset
