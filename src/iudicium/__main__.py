"""Main script to evaluate LLM translation of the Swiss constitution.

1. parse the xml consitution both english / romansh

IN: consitution.xml
OUT: dict[str, list[str]] where key is article number and value is list of paragraphs (pickled)
CHECKS: consitency betwen both languages in number of paragraphs and articles

2. for each paragraphs in each articles:

prompt llm being tested (an async function) to translate the paragaprh into romansh

IN: dict[str, list[str]] pargraph in english (pickled)
OUT: dict[str, list[str]] pargraph in romansh (pickled)
CHECKS: consitency betwen both languages in number of paragraphs and articles

Use async to parallelize the calls, maybe implement retry / rate limit?

3. evaluate with ROUGE / BLEU score the translation
"""

import argparse
import importlib
import json
import logging
import os
import sys
from datetime import datetime
from typing import NoReturn
from zoneinfo import ZoneInfo

from iudicium.metrics import compute_metrics
from iudicium.parser import parse
from iudicium.translators import TRANSLATORS, TranslatorProtocol

log = logging.getLogger(__name__)

tz = ZoneInfo("Europe/Zurich")


def try_import_translator(translator_name: str) -> type[TranslatorProtocol] | None:
    """Try to import translator and return class if successful."""
    try:
        module = importlib.import_module(f"iudicium.translators.{translator_name}")
        return getattr(module, "Translator")
    except ImportError as e:
        log.warning(
            f"Translator '{translator_name}' is not available due to missing dependencies: {e}"
        )
        log.warning(
            f"To use '{translator_name}', install with: uv add 'iudicium[{translator_name}]'"
        )
        return None
    except AttributeError as e:
        log.warning(
            f"Translator '{translator_name}' module found but missing Translator class: {e}"
        )
        return None
    except KeyError as e:
        log.warning(
            f"Translator '{translator_name}' is available but missing environment variable: {e}"
        )
        log.warning(
            f"Please check the documentation for '{translator_name}' translator setup"
        )
        return None
    except Exception as e:
        log.warning(f"Translator '{translator_name}' failed to load: {e}")
        return None


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM translation of the Swiss constitution."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level",
    )

    # Create subparsers for different translators
    subparsers = parser.add_subparsers(
        dest="translator",
        required=True,
        help="Choose the translator to use",
    )

    # Track available translators
    available_translators = []

    # Dynamically add arguments for each translator
    for translator_name in TRANSLATORS:
        translator_class = try_import_translator(translator_name)

        if translator_class is not None:
            # Only create subparser if translator is available
            translator_parser = subparsers.add_parser(
                translator_name,
                help=f"Use {translator_name} translator",
            )
            available_translators.append(translator_name)

            # Add translator-specific arguments
            for arg_names, arg_kwargs in translator_class.get_argparse_args():
                translator_parser.add_argument(*arg_names, **arg_kwargs)

    # Check if any translators are available
    if not available_translators:
        log.error(
            "No translators are available. Please install dependencies for at least one translator."
        )
        sys.exit(1)

    return parser


def main() -> NoReturn:
    """Main function to evaluate LLM translation of the Swiss constitution."""
    import asyncio

    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Initialize the selected translator
    try:
        # Re-import the translator class (we know it worked earlier)
        translator_class = try_import_translator(args.translator)
        if translator_class is None:
            log.error(f"Translator '{args.translator}' is not available")
            sys.exit(1)

        # create the translator from cli args provided.
        translator = translator_class.from_args(args)

    except ValueError as e:
        log.error(f"Invalid arguments: {e}")
        sys.exit(1)
    except KeyError as e:
        log.error(f"Missing required environment variable: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Failed to initialize translator: {e}")
        sys.exit(1)

    # parse and remove inconcistent articles in both languages
    log.info("Step 1. Parsing constitution xml files.")
    en_articles, rm_articles = parse(
        [
            "data/sources/SR-101-03032024-EN.xml",
            "data/sources/SR-101-03032024-RM.xml",
        ]
    )

    # use desired translator here
    # TODO: cache on disk the results for the same set of arguments
    log.info(f"Step 2. Calling {args.translator} translator.")
    translated_articles = asyncio.run(translator.translate(en_articles))

    # save the translated articles to a file
    dt = datetime.now(tz)
    os.makedirs(f"data/translations", exist_ok=True)
    with open(
        f"data/translations/{dt.isoformat()}_{translator.cache_key}.json", "w"
    ) as f:
        json.dump(translated_articles, f)

    log.info("Step 3. Assessing translated articles.")
    metrics = compute_metrics(translated_articles, rm_articles)
    print(metrics)

    # append values to results.csv
    with open("data/results.csv", "a") as f:
        f.write(
            f"{dt.isoformat()}, {translator.cache_key}, {metrics['bleu']}, {metrics['rouge']}\n"
        )

    # table output in terminal
    # csv file (each row = 1 paragraph + Average or total, each column = one metric)
    # TODO: Implement metrics module
    # metrics = metrics.compute(translated_articles, rm_articles)
    # print(metrics) # pretty table
    # metrics.write_csv(f"data/metrics/{dt.isoformat()}_{args.translator}_{sorted_args_passed}.csv")


if __name__ == "__main__":
    main()
