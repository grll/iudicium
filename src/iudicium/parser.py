"""Parser for the Swiss Constitution XML files."""

import logging
import xml.etree.ElementTree as ET

log = logging.getLogger(__name__)

INCONSISTENT_ARTICLES = {"83", "175", "189", "197"}
"""Articles that are not consistent across FR and RM versions:
 - Article 83: EN=1 vs RM=2 (+1 in Romansh)
 - Article 175: EN=4 vs RM=2 (-2 in Romansh)
 - Article 189: EN=4 vs RM=5 (+1 in Romansh)
 - Article 197: EN=26 vs RM=29 (+3 in Romansh)

details in `notebooks/001_check_parsing.ipynb`.
"""


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


def parse(xml_paths: list[str]) -> list[dict[str, list[str]]]:
    """Parse multiple constitution XML files and make sure they are consistent.

    Args:
        xml_paths (list[str]): List of paths to XML files.

    Returns:
        list[dict[str, list[str]]]: A list of dictionaries for each XML file.
    """
    parsed_articles = [parse_constitution(path) for path in xml_paths]

    log.info(
        f"Removing articles `{INCONSISTENT_ARTICLES}` that are not consistent across languages."
    )
    for art in INCONSISTENT_ARTICLES:
        for articles in parsed_articles:
            if art in articles:
                del articles[art]

    # ensure consistency before moving forward..
    assert all(
        set(articles.keys()) == set(parsed_articles[0].keys())
        for articles in parsed_articles[1:]
    ), "Article numbers do not match"

    assert all(
        len(articles[art_num]) == len(parsed_articles[0][art_num])
        for art_num in parsed_articles[0].keys()
        for articles in parsed_articles[1:]
    ), "Number of paragraphs do not match"

    log.info(f"Found {len(parsed_articles[0])} consistent articles.")
    log.info(
        f"Total consistent paragraphs: {sum(len(p) for p in parsed_articles[0].values())}"
    )
    return parsed_articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # parse and save constitution in both languages
    en_articles = parse_constitution("data/sources/SR-101-03032024-EN.xml")
    rm_articles = parse_constitution("data/sources/SR-101-03032024-RM.xml")

    for art in INCONSISTENT_ARTICLES:
        if art in en_articles:
            del en_articles[art]
        if art in rm_articles:
            del rm_articles[art]

    # ensure consistency before moving forward..
    assert set(en_articles.keys()) == set(rm_articles.keys()), (
        "Article numbers do not match"
    )
    for art_num in en_articles.keys():
        assert len(en_articles[art_num]) == len(rm_articles[art_num]), (
            f"Number of paragraphs do not match in article {art_num}"
        )

    # try the parse function as well
    articles = parse(
        [
            "data/sources/SR-101-03032024-EN.xml",
            "data/sources/SR-101-03032024-RM.xml",
        ]
    )
    assert articles[0] == en_articles
    assert articles[1] == rm_articles
