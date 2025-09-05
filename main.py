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
import pickle
import xml.etree.ElementTree as ET

log = logging.getLogger(__name__)

def parse_constitution(xml_path: str) -> dict[str, list[str]]:
    """Parse the constitution XML file and extract articles and their paragraphs.

    Args:
        xml_path (str): Path to the XML file.

    Returns:
        dict[str, list[str]]: A dictionary where keys are article numbers and values are lists of paragraphs.
    """
    log.info(f"Parsing constitution from '{xml_path}'")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {"akn": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}

    articles = {}

    for article in root.findall(".//akn:article", ns):
        article_id = article.get("eId")
        if not article_id:
            continue

        art_num = article_id.replace("art_", "")

        paragraphs = []
        for para in article.findall(".//akn:paragraph", ns):
            content = para.find(".//akn:p", ns)
            if content is not None and content.text:
                text = "".join(content.itertext()).strip()
                if text:
                    paragraphs.append(text)

        if not paragraphs:
            for content in article.findall(".//akn:content//akn:p", ns):
                if content is not None:
                    text = "".join(content.itertext()).strip()
                    if text:
                        paragraphs.append(text)

        if paragraphs:
            articles[art_num] = paragraphs

    log.info(f"Found {len(articles)} articles in the constitution")
    log.info(f"Total paragraphs: {sum(len(p) for p in articles.values())}")
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # parse and save constitution in both languages
    en_articles = parse_constitution("data/sources/SR-101-03032024-EN.xml")
    with open("data/processed/constitution_en.pkl", "wb") as f:
        pickle.dump(en_articles, f)

    rm_articles = parse_constitution("data/sources/SR-101-03032024-RM.xml")
    with open("data/processed/constitution_rm.pkl", "wb") as f:
        pickle.dump(rm_articles, f)

