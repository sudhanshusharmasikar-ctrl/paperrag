"""
Chunking: text cleaning, the noise filter, packing blocks into chunks, and the
page number every chunk carries. The last two tests describe known bugs.
"""
import pymupdf as fitz
import pytest

from app.chunking import _is_noise, _pack, chunk_pdf, clean, page_blocks
from app.config import CHUNK_CHARS, CHUNK_OVERLAP, MIN_CHUNK_CHARS

PARA = (
    "Retrieval augmented generation looks up relevant passages first and then "
    "asks the language model to answer using only those passages."
)


def test_clean_fixes_ligatures_and_curly_quotes():
    assert clean("ﬁne-tuning “works”") == 'fine-tuning "works"'


def test_clean_joins_a_word_split_across_lines():
    assert clean("multi-\nmodal models") == "multimodal models"


def test_clean_collapses_line_breaks_and_spaces():
    assert clean("  first line\n\n second   line  ") == "first line second line"


@pytest.mark.parametrize("text", ["12", "Table 3", "0.91 0.88 0.93 0.87 0.90 0.85 0.92"])
def test_noise_filter_drops_page_numbers_labels_and_table_rows(text):
    assert _is_noise(text)


def test_noise_filter_keeps_real_sentences():
    assert not _is_noise("Attention lets every token look at every other token.")


def test_pack_respects_the_size_limit_and_overlaps_neighbours():
    blocks = [f"Paragraph {i} " + "word " * 60 for i in range(6)]  # ~310 chars each
    chunks = list(_pack(blocks))
    assert len(chunks) > 1
    assert all(len(c) <= CHUNK_CHARS for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        # each chunk starts with the last CHUNK_OVERLAP characters of the one before
        assert nxt.startswith(prev[-CHUNK_OVERLAP:])


def test_pack_splits_an_oversized_paragraph_at_sentence_ends():
    block = " ".join(f"Sentence {i} is about ranking quality." for i in range(60))
    assert len(block) > CHUNK_CHARS
    chunks = list(_pack([block]))
    assert len(chunks) > 1
    assert all(len(c) <= CHUNK_CHARS for c in chunks)
    assert all(c.endswith(".") for c in chunks)


def test_chunk_pdf_gives_each_chunk_exactly_one_page(make_pdf):
    path = make_pdf("paper.pdf", [[f"Alpha page. {PARA}"], [f"Omega page. {PARA}"]])
    chunks = chunk_pdf(path)
    assert [(c.chunk_id, c.page) for c in chunks] == [("paper::p1::c0", 1), ("paper::p2::c0", 2)]
    assert all(c.source == "paper.pdf" and c.doc_id == "paper" for c in chunks)
    assert "Omega" not in chunks[0].text and "Alpha" not in chunks[1].text


def test_chunk_pdf_skips_pages_without_usable_text(make_pdf):
    caption = "A caption that is far too short to be a chunk."  # survives the noise filter
    assert len(caption) < MIN_CHUNK_CHARS
    path = make_pdf("paper.pdf", [[], [caption], [PARA]])  # blank, too short, real text
    assert [c.page for c in chunk_pdf(path)] == [3]


# ------------------------------------------------------------------ known bugs
# xfail = "expected to fail": these tests describe bugs that are not fixed yet.
# strict=True turns an unexpected pass into a failure, so whoever fixes the bug
# has to remove the marker in the same change.


@pytest.mark.xfail(strict=True, reason="known bug: page_blocks() sorts by y, then x, "
                   "which interleaves the two columns of a page")
def test_two_column_page_is_read_column_by_column(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    filler = "words that make this paragraph long enough to survive the noise filter"

    def put(x0, y0, x1, label):
        page.insert_textbox(fitz.Rect(x0, y0, x1, y0 + 60), f"{label} paragraph, {filler}.",
                            fontsize=10)

    put(60, 50, 552, "TITLE")             # spans both columns
    for i, y in enumerate([120, 220, 320], start=1):
        put(60, y, 296, f"L{i}")           # left column
        put(316, y + 52, 552, f"R{i}")     # right column, starting a little lower
    path = tmp_path / "twocol.pdf"
    doc.save(path)
    doc.close()

    with fitz.open(path) as pdf:
        order = [block.split()[0] for block in page_blocks(pdf[0])]
    assert order == ["TITLE", "L1", "L2", "L3", "R1", "R2", "R3"]


@pytest.mark.xfail(strict=True, reason="known bug: the overlap carried into a new chunk "
                   "isn't counted, so a long paragraph can push a chunk past CHUNK_CHARS")
def test_overlap_never_pushes_a_chunk_past_the_limit():
    blocks = ["first " * 80, "second " * 120]  # ~480 and ~840 characters
    assert all(len(c) <= CHUNK_CHARS for c in _pack(blocks))
