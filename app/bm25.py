"""
Keyword search with BM25, the other half of hybrid search.

The embedding model matches meaning, so it finds a passage that says the same
thing in other words. It is weaker at exact terms: a name like "WordPiece", a
number like "30,000", an acronym. BM25 scores a passage by which of the
question's words it contains and weights each word by how rare it is in the
whole library, so a rare term that appears in only a few passages counts for
much more than "model" or "training".

For each question word w in passage d:

    idf(w) * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / average length))

f is how often w occurs in d. k1 makes repeats count less and less (the 10th
"attention" adds little), and b stops a long passage from winning just by
containing more words. k1 = 1.5 and b = 0.75 are the usual defaults.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

# Words too common to say anything about a passage. BM25's idf already gives
# them almost no weight; dropping them keeps the postings lists short.
STOPWORDS = set(
    "a an and are as at be by can did do does for from has have how if in into "
    "is it its of on or that the their them they this to was we were what when "
    "where which who why will with".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers: "LAION-5B uses 30,000 WordPieces" ->
    ["laion", "5b", "uses", "30", "000", "wordpieces"]."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(texts)
        # postings: for every word, the passages that contain it and how often
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = np.zeros(self.n)
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.lengths[i] = sum(counts.values())
            for word, f in counts.items():
                self.postings[word].append((i, f))
        self.avg_length = float(self.lengths.mean()) if self.n else 0.0
        # rare words weigh more; this form of idf is never negative
        self.idf = {w: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
                    for w, p in self.postings.items()}

    def scores(self, query: str) -> np.ndarray:
        """One score per passage; 0 for a passage with none of the words."""
        out = np.zeros(self.n)
        if not self.n:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avg_length, 1e-9))
        for word in set(tokenize(query)):
            for i, f in self.postings.get(word, ()):
                out[i] += self.idf[word] * f * (self.k1 + 1) / (f + norm[i])
        return out
