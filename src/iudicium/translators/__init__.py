"""Translators package for converting Swiss constitution text to Romansh.

This package contains various translator implementations that receive a dictionary
of articles containing paragraphs of the Swiss constitution in English and translate
them into the same data structure but in Romansh.

The module defines a Protocol that each translator implementation must follow to
ensure consistent behavior when swapping translators in the main module.
"""
from typing import Protocol

TRANSLATORS = ["apertus", "openrouter"]
"""list of possible translator values (match modules in `translators` package)"""

class TranslatorProtocol(Protocol):
    """Protocol for translator implementations."""

    async def translate(self, articles: dict[str, list[str]]) -> dict[str, list[str]]:
        """Translate the articles from English to Romansh."""
        ...