from importlib.resources import files
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from rank_llm.data import Result
from rank_llm.rerank.pointwise.pointwise_rankllm import PointwiseRankLLM

TEMPLATES = files("rank_llm.rerank.prompt_templates")


class GaggleReranker(PointwiseRankLLM):
    def __init__(
        self,
        model: str = "castorini/gaggle-reranker-20261005",
        prompt_template_path: str = TEMPLATES / "gaggle_template.yaml",
        device: str = "cuda",
        batch_size: int = 16,
        backend: str = "transformers",
    ):
        if (
            Path(prompt_template_path).resolve()
            != Path(TEMPLATES / "gaggle_template.yaml").resolve()
        ):
            raise ValueError("Gaggle rerankers use a fixed prompt template")
        super().__init__(
            model=model,
            context_size=2048,
            prompt_template_path=prompt_template_path,
            device=device,
            batch_size=batch_size,
        )
        if backend not in {"transformers", "modal"}:
            raise ValueError(f"Unknown Gaggle backend: {backend}")
        self._modal_handler = None
        self._config = None
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if backend == "modal":
            from rank_llm.rerank.modal_handler import ModalHandler

            self._config = AutoConfig.from_pretrained(model, trust_remote_code=True)
            self._modal_handler = ModalHandler(model, batch_size)
            self._llm = None
            return

        dtype = torch.float32
        if device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._llm = (
            AutoModelForSequenceClassification.from_pretrained(
                model, trust_remote_code=True, torch_dtype=dtype
            )
            .to(device)
            .eval()
        )

    def run_llm_batched(
        self, prompts: list[dict[str, str]]
    ) -> tuple[list[str], list[int], list[float]]:
        if self._modal_handler is not None:
            return self._modal_handler.run_llm_batched(prompts)
        pairs = [(prompt["query"], prompt["passage"]) for prompt in prompts]
        scores = self._llm.compute_score(
            pairs, tokenizer=self._tokenizer, batch_size=self._batch_size
        )
        return [str(score) for score in scores], [0] * len(scores), scores

    def run_llm(self, prompt: dict[str, str]) -> tuple[str, int, float]:
        outputs, counts, scores = self.run_llm_batched([prompt])
        return outputs[0], counts[0], scores[0]

    def create_prompt(self, result: Result, index: int) -> tuple[dict[str, str], int]:
        doc = result.candidates[index].doc
        passage = next(
            (
                doc[key]
                for key in ("text", "segment", "contents", "content", "body", "passage")
                if key in doc
            ),
            None,
        )
        if passage is None:
            raise ValueError("Candidate document has no passage text")
        if doc.get("title"):
            passage = f"Title: {doc['title']} Content: {passage}"
        prompt = {"query": result.query.text, "passage": passage}
        return prompt, self.get_num_tokens(prompt)

    def get_num_tokens(self, prompt: dict[str, str]) -> int:
        if self._config is not None:
            config = self._config
            query = self._tokenizer.encode(
                prompt["query"], add_special_tokens=False, split_special_tokens=True
            )[: config.max_query_len]
            passage = self._tokenizer.encode(
                prompt["passage"], add_special_tokens=False, split_special_tokens=True
            )[: config.max_doc_len]
            return (
                len(config.prompt_prefix_ids)
                + len(query)
                + len(config.prompt_mid_ids)
                + len(passage)
                + len(config.prompt_suffix_ids)
            )
        return len(
            self._llm.build_input_ids(
                self._tokenizer, prompt["query"], prompt["passage"]
            )
        )

    def num_output_tokens(self) -> int:
        return 0

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0
