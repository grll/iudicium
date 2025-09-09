"""Translator using OpenRouter.

This module provides a Translator implementation using the OpenRouter API.

In order to use this translator, you need to set the OPENROUTER_API_KEY environment variable.
You can get your API key from OpenRouter settings: https://openrouter.ai/settings/keys.

The module will try to load the api key from the .env file at import time.

Run `uv add iudicium[openrouter]` to install the necessary dependencies.
"""

import asyncio
import logging
import os
from textwrap import dedent
from typing import Literal

from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from iudicium.cache import PartialTranslationsError, cached_translation
from iudicium.translators import TranslatorProtocol

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        "Some necessary packages are not installed for OpenRouter Translator. "
        "Please install them with `uv add iudicium[openrouter]`."
    ) from e


log = logging.getLogger(__name__)


load_dotenv()

# fail as early as possible if open router API key is not set.
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
"""OpenRouter API KEY usually set in `.env`. Get yours: https://openrouter.ai/settings/keys."""

OPENROUTER_MODELS = Literal["openai/gpt-5-nano"]
"""Models available on OpenRouter."""

PROMPT = """
Translate the following text from English to Romansh.

Text to translate:
{text}

Return **ONLY** the translated text, without any additional commentary.
""".strip()
"""User Completion Prompt for translating English text to Romansh."""


class Translator(TranslatorProtocol):
    """Translator using OpenRouter."""

    @property
    def cache_key(self) -> str:
        """Generate cache key for this configuration."""
        model_slug = self.model.replace("/", "_").replace("-", "")
        return f"openrouter_{model_slug}_{self.temperature}"

    def __init__(
        self,
        model: OPENROUTER_MODELS = "openai/gpt-5-nano",
        temperature: float = 0.7,
        concurrency: int = 10,
    ):
        """Initialize the OpenRouterTranslator.

        Args:
            model: the model to use for translation.
            temperature: the temperature to use for translation.
            concurrency: the number of concurrent requests to make.
        """
        self.model = model
        self.temperature = temperature
        self.concurrency = concurrency

        api_key = os.environ["OPENROUTER_API_KEY"]
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    @classmethod
    def get_argparse_args(cls) -> list[tuple[list[str], dict]]:
        """Return argparse argument definitions for OpenRouter translator.

        Returns:
            List of tuples with argument definitions.
        """
        return [
            (
                ["--model", "-m"],
                {
                    "type": str,
                    "default": "openai/gpt-5-nano",
                    "help": "OpenRouter model to use for translation. See https://openrouter.ai/models for available models. Default to gpt-5-nano.",
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
                ["--concurrency", "-c"],
                {
                    "type": int,
                    "default": 10,
                    "help": "Number of concurrent API requests. Default to 10.",
                },
            ),
        ]

    @classmethod
    def from_args(cls, args) -> "Translator":
        """Create OpenRouter translator from parsed arguments.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Configured OpenRouter translator instance.

        Raises:
            ValueError: If arguments are invalid.
        """
        # Validate temperature range
        if not 0.0 <= args.temperature <= 2.0:
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0, got {args.temperature}"
            )

        # Validate concurrency
        if args.concurrency < 1:
            raise ValueError(f"Concurrency must be at least 1, got {args.concurrency}")

        return cls(
            model=args.model,
            temperature=args.temperature,
            concurrency=args.concurrency,
        )

    async def _translate_paragraph(
        self,
        paragraph: str,
        semaphore: asyncio.Semaphore | None = None,
        progress_bar: tqdm | None = None,
    ) -> str:
        """Translate a single paragraph using the client."""

        async def _call_api():
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": PROMPT.format(text=paragraph)},
                ],
                model=self.model,
                temperature=self.temperature,
            )
            # TODO: handle errors / retries
            content = response.choices[0].message.content
            if progress_bar:
                progress_bar.update(1)
            return content

        if semaphore:
            async with semaphore:
                return await _call_api()
        else:
            return await _call_api()

    @cached_translation
    async def translate(
        self,
        articles: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Translate the articles from English to Romansh using OpenRouter.

        Args:
            articles: a dict containing all the articles of the consitution.

        Returns:
            a dict containing all the articles of the consitution in Romansh.
        """
        log.info(f"translating {self.concurrency} paragraph at a time.")

        # task_dict holds the tasks for each article
        task_dict: dict[str, list[asyncio.Task[str]]] = {}
        semaphore = asyncio.Semaphore(self.concurrency)
        progress_bar = tqdm(
            total=sum(len(paragraphs) for paragraphs in articles.values()),
            desc="Translating",
        )

        # TODO: handle errors / retries
        try:
            with logging_redirect_tqdm():
                async with asyncio.TaskGroup() as tg:
                    for article, paragraphs in articles.items():
                        task_dict[article] = [
                            tg.create_task(
                                self._translate_paragraph(
                                    paragraph, semaphore, progress_bar
                                )
                            )
                            for paragraph in paragraphs
                        ]
        except BaseException as e:
            # Persist only fully-completed articles to avoid cache/pending mismatches.
            partial: dict[str, list[str]] = {}
            for article, tasks in task_dict.items():
                # all paragraphs done, not cancelled, and no exception
                if all(
                    t.done() and not t.cancelled() and t.exception() is None
                    for t in tasks
                ):
                    partial[article] = [t.result() for t in tasks]
            # Surface partials to the caching decorator.
            raise PartialTranslationsError(partial) from e
        finally:
            progress_bar.close()

        # reconstruct the result with the same structure
        translated_articles: dict[str, list[str]] = {}
        for article, tasks in task_dict.items():
            translated_articles[article] = [task.result() for task in tasks]

        return translated_articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    openrouter_translator = Translator()

    # test translating a single paragraph
    paragraph = "The Confederation shall promote scientific research and innovation."
    translation = asyncio.run(openrouter_translator._translate_paragraph(paragraph))
    log.info(
        dedent(
            f"""Original (EN): {paragraph}
                Translation (RM): {translation}
            """.strip()
        )
    )

    # test translating multiple articles
    articles = {
        "64": [
            "The Confederation shall promote scientific research and innovation.",
            "It may make its support conditional in particular on quality assurance and coordination being guaranteed.",
            "It may establish, take over or run research institutes.",
        ]
    }
    translated_articles = asyncio.run(openrouter_translator.translate(articles))
    log.info(translated_articles)
