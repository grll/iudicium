"""Translator using OpenRouter."""

import asyncio
import logging
import os
from textwrap import dedent
from typing import Literal

from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

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

    async def translate(
        self,
        articles: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Translate the articles from English to Romansh using OpenRouter.

        Args:
            articles: a dict containing all the articles of the consitution.
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
