"""Caching utilities for translators.

This module provides a decorator for caching translation results to disk.
Cache files are stored in data/cache/ and are keyed by translator configuration
and input data hash.
"""

import functools
import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
"""Directory to store cache files."""

# create cache directory if it doesn't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_data_hash(articles: dict[str, list[str]]) -> str:
    """Compute the hash of the input data.

    Args:
        articles: Dictionary of articles
    """
    return hashlib.sha256(json.dumps(articles, sort_keys=True).encode()).hexdigest()[:8]


def get_cache_file(cache_key: str, data_hash: str) -> Path:
    """Get the path to the cache file.

    Args:
        cache_key: Translator configuration key
        data_hash: Hash of input data
    """
    return CACHE_DIR / f"{cache_key}_{data_hash}.json"


def save_cache(cache_key: str, data_hash: str, completed: dict[str, list[str]]) -> None:
    """Save translation cache to disk.

    Args:
        cache_file: Path to the cache file
        cache_key: Translator configuration key
        data_hash: Hash of input data
        completed: Dictionary of completed translations
    """
    cache_file = get_cache_file(cache_key, data_hash)
    try:
        with open(cache_file, "w") as f:
            json.dump(
                {
                    "cache_key": cache_key,
                    "data_hash": data_hash,
                    "completed": completed,
                    "num_articles": len(completed),
                    "num_paragraphs": sum(len(paras) for paras in completed.values()),
                },
                f,
                indent=2,
            )
    except Exception as e:
        log.error(f"Failed to save cache to {cache_file}: {e}")


def load_cache(cache_key: str, data_hash: str) -> dict[str, list[str]]:
    """Load cache from disk.

    Args:
        cache_key: Translator configuration key
        data_hash: Hash of input data

    Returns:
        Dictionary of completed translations
    """
    # Load existing cache if available
    cache_file = get_cache_file(cache_key, data_hash)
    completed = {}
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cache_data = json.load(f)
                completed = cache_data.get("completed", {})
                log.info(f"Loaded {len(completed)} cached translations")
        except Exception as e:
            log.warning(f"Failed to load cache from {cache_file}: {e}")
    return completed


class PartialTranslationsError(Exception):
    """Carries fully-completed article translations produced before a failure."""

    def __init__(self, partial: dict[str, list[str]], message: str | None = None):
        """Initialize the PartialTranslationsError.

        Args:
            partial: Dictionary of partially completed translations
            message: Message to display
        """
        super().__init__(message or "Partial translations available")
        self.partial = partial


def cached_translation(method):
    """Decorator for caching translation results.

    Expects the decorated method to be on a class with a `cache_key` property.
    Automatically loads cached results and saves on completion or error.

    The decorator:
    1. Checks for existing cached translations
    2. Only passes uncached articles to the wrapped method
    3. Merges results with cached data
    4. Saves complete results on success or partial results on failure

    Args:
        method: An async translate method that takes articles dict and returns translations

    Returns:
        Wrapped method with caching functionality
    """

    @functools.wraps(method)
    async def wrapper(self, articles: dict[str, list[str]]) -> dict[str, list[str]]:
        # Generate cache filename based on translator config and input data
        data_hash = compute_data_hash(articles)

        completed = load_cache(self.cache_key, data_hash)

        # Determine which articles still need translation
        pending = {k: v for k, v in articles.items() if k not in completed}

        if not pending:
            log.info("All articles found in cache, returning cached results")
            return completed

        log.info(
            f"Translating {len(pending)} articles ({len(completed)} already cached)"
        )

        try:
            # Call the original translate method with only pending articles
            newly_translated = await method.__get__(self, type(self))(pending)

            # Merge with cached results
            result = {**completed, **newly_translated}

            # Save complete results to cache
            save_cache(self.cache_key, data_hash, result)
            log.info(f"Successfully cached {len(result)} total translations")

            return result

        except BaseException as e:  # catch ExceptionGroup too (Py 3.11+)
            # If translate() surfaced partial results, merge them with what we had.
            partial = getattr(e, "partial", None)
            to_save = {**completed, **(partial or {})}
            if to_save:
                save_cache(self.cache_key, data_hash, to_save)
                new_items = len(to_save) - len(completed)
                if new_items > 0:
                    log.info(
                        f"Saved {len(to_save)} translations to cache after error ({new_items} new)"
                    )
                else:
                    log.info(f"Saved {len(completed)} cached translations after error")
            log.error(f"Translation failed: {e}")
            raise

    return wrapper


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_cache(
        "openrouter_gpt5nano_0.7",
        "12345678",
        {
            "article1": ["paragraph1", "paragraph2"],
            "article2": ["paragraph3", "paragraph4"],
        },
    )
    completed = load_cache("openrouter_gpt5nano_0.7", "12345678")
    log.info(completed)

    import asyncio

    class _TestTranslator:
        @property
        def cache_key(self) -> str:
            return "test_translator_1.0"

        @cached_translation
        async def translate(
            self, articles: dict[str, list[str]]
        ) -> dict[str, list[str]]:
            # Simulate translation
            return {k: [f"translated: {p}" for p in v] for k, v in articles.items()}

    translator = _TestTranslator()
    result = asyncio.run(
        translator.translate(
            {
                "article1": ["paragraph1", "paragraph2"],
                "article2": ["paragraph3", "paragraph4"],
            }
        )
    )
    log.info(f"Test result: {result}")
