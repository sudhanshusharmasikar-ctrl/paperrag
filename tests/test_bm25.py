"""BM25 keyword search: rare words count most, repeats and long passages less."""
from app.bm25 import BM25, tokenize


def test_tokenize_keeps_words_and_numbers():
    assert tokenize("LAION-5B uses 30,000 WordPieces in the model.") == \
        ["laion", "5b", "uses", "30", "000", "wordpieces", "model"]


def test_a_rare_word_outweighs_a_common_one():
    texts = ["the model is trained on text", "the model uses wordpiece tokens",
             "a model of images", "model model model"]
    scores = BM25(texts).scores("wordpiece model")
    assert scores.argmax() == 1  # the only passage with the rare word
    assert scores[1] > 3 * max(scores[0], scores[2], scores[3])


def test_a_passage_without_the_words_scores_zero():
    scores = BM25(["attention is all you need", "convolutional networks"]).scores("attention heads")
    assert scores[0] > 0 and scores[1] == 0


def test_repeating_a_word_counts_less_and_less():
    texts = ["dropout rate", "dropout " * 10 + "rate", "other words here", "more other words"]
    once, ten = BM25(texts).scores("dropout")[:2]
    assert once < ten < 2 * once  # ten times the word, not ten times the score


def test_a_shorter_passage_wins_on_the_same_count():
    texts = ["warmup steps", "warmup steps and many other words about the training schedule",
             "unrelated text"]
    scores = BM25(texts).scores("warmup")
    assert scores[0] > scores[1] > 0


def test_an_empty_library_or_a_question_of_stopwords_scores_nothing():
    assert BM25([]).scores("anything").shape == (0,)
    assert not BM25(["some text"]).scores("what is the").any()
