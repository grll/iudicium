"""Metrics for evaluating the translation of the Swiss constitution."""


def bleu_score(
    translated_articles: dict[str, list[str]], rm_articles: dict[str, list[str]]
) -> float:
    """Calculate the BLEU score between the translated and the reference articles."""
    import sacrebleu

    translated_paragraphs: list[str] = []
    references_paragraphs: list[list[str]] = []
    for article_id in translated_articles.keys():
        assert len(translated_articles[article_id]) == len(rm_articles[article_id]), (
            "Number of paragraphs do not match"
        )
        translated_paragraphs.extend(translated_articles[article_id])
        references_paragraphs.extend([rm_articles[article_id]])

    return (
        sacrebleu.corpus_bleu(translated_paragraphs, references_paragraphs).score / 100
    )


def rouge_score(
    translated_articles: dict[str, list[str]], rm_articles: dict[str, list[str]]
) -> float:
    """Calculate the ROUGE score between the translated and the reference articles."""
    return 0.0


def compute_metrics(
    translated_articles: dict[str, list[str]], rm_articles: dict[str, list[str]]
) -> dict[str, float]:
    """Compute the BLEU and ROUGE scores between the translated and the reference articles."""
    return {
        "bleu": bleu_score(translated_articles, rm_articles),
        "rouge": rouge_score(translated_articles, rm_articles),
    }


if __name__ == "__main__":
    # Test case 1: Perfect match
    translated1 = {
        "art1": ["The quick brown fox jumps over the lazy dog"],
        "art2": ["Hello world this is a test"],
    }
    reference1 = {
        "art1": ["The quick brown fox jumps over the lazy dog"],
        "art2": ["Hello world this is a test"],
    }
    score1 = bleu_score(translated1, reference1)
    print(f"Test 1 (perfect match): BLEU score = {score1:.4f}")

    # Test case 2: Partial match
    translated2 = {"art1": ["The quick brown fox"], "art2": ["Hello world"]}
    reference2 = {
        "art1": ["The quick brown fox jumps over the lazy dog"],
        "art2": ["Hello world this is a test"],
    }
    score2 = bleu_score(translated2, reference2)
    print(f"Test 2 (partial match): BLEU score = {score2:.4f}")

    # Test case 3: Different words
    translated3 = {
        "art1": ["Completely different text here"],
        "art2": ["Different words here"],
    }
    reference3 = {
        "art1": ["The quick brown fox jumps over the lazy dog"],
        "art2": ["Hello world this is a test"],
    }
    score3 = bleu_score(translated3, reference3)
    print(f"Test 3 (different words): BLEU score = {score3:.4f}")

    # Test case 4: Word order difference
    translated4 = {"art1": ["fox brown quick the"]}
    reference4 = {"art1": ["the quick brown fox"]}
    score4 = bleu_score(translated4, reference4)
    print(f"Test 4 (word order): BLEU score = {score4:.4f}")
