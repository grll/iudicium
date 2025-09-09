"""Translator using Transformers library with local models.

This module provides a Translator implementation using HuggingFace Transformers.

Run `uv add iudicium[transformers]` to install the necessary dependencies.
"""

import asyncio
import logging
import os
from typing import Literal

from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from iudicium.translators import TranslatorProtocol

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    raise ImportError(
        "Some necessary packages are not installed for Transformers Translator. "
        "Please install them with `uv add iudicium[transformers]` or `uv add iudicium[transformers-cpu]` or `uv add iudicium[transformers-cu129]`."
    ) from e


log = logging.getLogger(__name__)

load_dotenv()

# fail as early as possible if HF Token is not set.
# also make sure you go on the model page and accept the terms of use to unlock gated models.
HF_TOKEN = os.environ["HF_TOKEN"]
"""HF Token usually set in `.env`. Get yours: https://huggingface.co/settings/tokens."""

PROMPT = """
Translate the following text from English to Romansh.

Text to translate:
{text}

Return **ONLY** the translated text, without any additional commentary.
""".strip()
"""User Completion Prompt for translating English text to Romansh."""


class Translator(TranslatorProtocol):
    """Translator using Transformers library."""

    @property
    def cache_key(self) -> str:
        """Generate cache key for this configuration."""
        model_slug = self.model_name.replace("/", "_").replace("-", "")
        return f"transformers_{model_slug}_{self.temperature}_{self.top_p}"

    def __init__(
        self,
        model_name: str = "swiss-ai/Apertus-8B-Instruct-2509",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 2048,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the Transformers Translator.

        Args:
            model_name: HuggingFace model name to use for translation.
            temperature: Temperature for text generation.
            top_p: Nucleus sampling parameter.
            max_new_tokens: Maximum number of new tokens to generate.
            batch_size: Number of paragraphs to process in parallel.
            device: Device to run the model on ("cuda" or "cpu").
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.device = device

        log.info(f"Loading model {model_name} on {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        # Some causal LMs don't define a pad token; use EOS as PAD to allow batch padding.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
        ).to(device)

    @classmethod
    def get_argparse_args(cls) -> list[tuple[list[str], dict]]:
        """Return argparse argument definitions for Transformers translator.

        Returns:
            List of tuples with argument definitions.
        """
        return [
            (
                ["--model-name", "-m"],
                {
                    "type": str,
                    "default": "swiss-ai/Apertus-8B-Instruct-2509",
                    "help": "HuggingFace model name to use for translation.",
                },
            ),
            (
                ["--temperature", "-t"],
                {
                    "type": float,
                    "default": 0.7,
                    "help": "Temperature for text generation (0.0-2.0). Default to 0.7.",
                },
            ),
            (
                ["--top-p"],
                {
                    "type": float,
                    "default": 0.9,
                    "help": "Nucleus sampling parameter (0.0-1.0). Default to 0.9.",
                },
            ),
            (
                ["--max-new-tokens"],
                {
                    "type": int,
                    "default": 2048,
                    "help": "Maximum number of new tokens to generate. Default to 2048.",
                },
            ),
            (
                ["--batch-size", "-b"],
                {
                    "type": int,
                    "default": 1,
                    "help": "Number of paragraphs to process in parallel. Default to 1.",
                },
            ),
            (
                ["--device", "-d"],
                {
                    "type": str,
                    "default": "cuda" if torch.cuda.is_available() else "cpu",
                    "choices": ["cuda", "cpu"],
                    "help": "Device to run the model on.",
                },
            ),
        ]

    @classmethod
    def from_args(cls, args) -> "Translator":
        """Create Transformers translator from parsed arguments.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Configured Transformers translator instance.

        Raises:
            ValueError: If arguments are invalid.
        """
        if not 0.0 <= args.temperature <= 2.0:
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0, got {args.temperature}"
            )

        if not 0.0 <= args.top_p <= 1.0:
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {args.top_p}")

        if args.batch_size < 1:
            raise ValueError(f"Batch size must be at least 1, got {args.batch_size}")

        if args.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be at least 1, got {args.max_new_tokens}"
            )

        return cls(
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            device=args.device,
        )

    def _translate_batch(self, paragraphs: list[str]) -> list[str]:
        """Translate a batch of paragraphs."""
        messages_list = [
            [{"role": "user", "content": PROMPT.format(text=paragraph)}]
            for paragraph in paragraphs
        ]

        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_list
        ]

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,  # be explicit
            )

        # Use attention_mask to recover the true (unpadded) input length per row
        input_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()

        results = []
        for i, in_len in enumerate(input_lengths):
            output_ids = generated_ids[i][in_len:]
            translated_text = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip()
            results.append(translated_text)

        return results

    async def translate(
        self,
        articles: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Translate the articles from English to Romansh using Transformers.

        Args:
            articles: a dict containing all the articles of the constitution.

        Returns:
            a dict containing all the articles of the constitution in Romansh.
        """
        log.info(f"Translating with batch size {self.batch_size}")

        # Prepare output buffers with correct sizes
        translated_articles: dict[str, list[str]] = {
            art_id: [None] * len(pars) for art_id, pars in articles.items()
        }

        # Flatten all (article_id, paragraph_index, text) for global batching
        flat: list[tuple[str, int, str]] = []
        for art_id, pars in articles.items():
            for idx, text in enumerate(pars):
                flat.append((art_id, idx, text))

        total_paragraphs = len(flat)
        progress_bar = tqdm(total=total_paragraphs, desc="Translating")

        try:
            with logging_redirect_tqdm():
                for i in range(0, total_paragraphs, self.batch_size):
                    batch_meta = flat[i : i + self.batch_size]
                    batch_texts = [t for _, _, t in batch_meta]

                    # Run the synchronous batch on a thread to keep the event loop responsive
                    translations = await asyncio.to_thread(
                        self._translate_batch, batch_texts
                    )

                    # Stitch results back in place
                    for (art_id, idx, _), tr in zip(batch_meta, translations):
                        translated_articles[art_id][idx] = tr

                    progress_bar.update(len(batch_meta))
        finally:
            progress_bar.close()

        return translated_articles


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    transformers_translator = Translator()

    articles = {
        "64": [
            "The Confederation shall promote scientific research and innovation.",
            "It may make its support conditional in particular on quality assurance and coordination being guaranteed.",
            "It may establish, take over or run research institutes.",
        ]
    }

    translated_articles = asyncio.run(transformers_translator.translate(articles))
    log.info(translated_articles)
