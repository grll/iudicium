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

import logging

from iudicium.parser import parse
from iudicium.translators import TRANSLATORS

log = logging.getLogger(__name__)



if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    # use argparse to parse cli args
    # verify arguments passed
    translator = "openrouter"
    match translator:
        case "openrouter":
            # model argument (default to gpt5-nano)
            # semaphore could be provided
            # temperature / other openai client options could be provided
            from iudicium.translators.openrouter import Translator
            translator = Translator()

        case "apertus":
            # 8B or 70B
            # batch size could be provided
            # other transformers options could be provided?
            from iudicium.translators.apertus import Translator
            translator = Translator()

        case default:
            raise AttributeError(f"Translator `{translator}` unkown. Supported translators: `{TRANSLATORS}`")


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
    log.info(f"Step 2. Calling translator.")
    translated_articles = asyncio.run(translator.translate(articles))
    
    log.info("Step 3. Assessing translated articles.")
    # table output in terminal
    # csv file (each row = 1 paragraph + Average or total, each column = one metric)
    metrics = metrics.compute(translated_articles, rm_articles)
    print(metrics) # pretty table
    metrics.write_csv("data/metrics/{dt.isoformat()}_{translator}_{sorted_args_passed}.csv")

