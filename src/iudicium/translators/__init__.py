"""Translators package for converting Swiss constitution text to Romansh.

This package contains various translator implementations that receive a dictionary
of articles containing paragraphs of the Swiss constitution in English and translate
them into the same data structure but in Romansh.

The module defines a Protocol that each translator implementation must follow to
ensure consistent behavior when swapping translators in the main module.
"""

import argparse
import os
import pkgutil
from typing import Protocol


def _discover_translators() -> list[str]:
    """Automatically discover translator modules in this package."""
    translators = []
    package_path = os.path.dirname(__file__)

    for _, module_name, is_pkg in pkgutil.iter_modules([package_path]):
        if not is_pkg and not module_name.startswith("_"):
            translators.append(module_name)

    return sorted(translators)


TRANSLATORS = _discover_translators()
"""list of possible translator values (automatically discovered from modules in the `translators` package)"""


class TranslatorProtocol(Protocol):
    """Protocol for translator implementations."""

    @property
    def cache_key(self) -> str:
        """Return a unique string identifying this translator configuration.

        Used for cache file naming. Should include all parameters that affect output.
        Example: "openrouter_gpt5nano_0.7"
        """
        ...

    @classmethod
    def get_argparse_args(cls) -> list[tuple[list[str], dict]]:
        """Return argparse argument definitions for this translator.

        Returns:
            List of tuples, each containing:
            - List of argument names (e.g., ['--model', '-m'])
            - Dict of argparse.add_argument() keyword arguments
        """
        ...

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TranslatorProtocol":
        """Create translator instance from parsed arguments.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Configured translator instance.

        Raises:
            ValueError: If arguments are invalid.
            KeyError: If required environment variables are missing.
        """
        ...

    async def translate(self, articles: dict[str, list[str]]) -> dict[str, list[str]]:
        """Translate the articles from English to Romansh."""
        ...
