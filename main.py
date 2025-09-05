"""
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