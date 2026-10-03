"""
Chunking: text cleaning, the noise filter, reading order on one- and two-column
pages, packing blocks into chunks, and the page number every chunk carries.
"""
import pymupdf as fitz
import pytest

from app import chunking
from app.chunking import _is_noise, _pack, chunk_pdf, clean, page_blocks
from app.config import CHUNK_CHARS, CHUNK_OVERLAP, MIN_CHUNK_CHARS

PARA = (
    "Retrieval augmented generation looks up relevant passages first and then "
    "asks the language model to answer using only those passages."
)


def test_clean_fixes_ligatures_and_curly_quotes():
    assert clean("\ufb01ne-tuning \u201cworks\u201d") == 'fine-tuning "works"'


def test_clean_joins_a_word_split_across_lines():
    assert clean("multi-\nmodal models") == "multimodal models"


def test_clean_collapses_line_breaks_and_spaces():
    assert clean("  first line\n\n second   line  ") == "first line second line"


@pytest.mark.parametrize("text", ["12", "Table 3", "0.91 0.88 0.93 0.87 0.90 0.85 0.92"])
def test_noise_filter_drops_page_numbers_labels_and_table_rows(text):
    assert _is_noise(text)


def test_noise_filter_keeps_real_sentences():
    assert not _is_noise("Attention lets every token look at every other token.")


# ---------------------------------------------------------------- reading order
FILLER = "words that make this paragraph long enough to survive the noise filter"


def para(label):
    return f"{label} paragraph, {FILLER}."


def reading_order(tmp_path, boxes):
    """Draw each (x0, y0, x1, text) box on one page and return the first word
    of every block, in the order page_blocks() reads them."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # the middle of the page is x = 306
    for x0, y0, x1, text in boxes:
        page.insert_textbox(fitz.Rect(x0, y0, x1, y0 + 60), text, fontsize=10)
    path = tmp_path / "layout.pdf"
    doc.save(path)
    doc.close()
    with fitz.open(path) as pdf:
        return [block.split()[0] for block in page_blocks(pdf[0])]


def test_one_column_page_is_read_top_to_bottom(tmp_path):
    boxes = [
        (60, 100, 552, para("A")),
        (60, 150, 552, "B is a short line on its own."),  # ends left of the middle
        (60, 200, 552, para("C")),
    ]
    assert reading_order(tmp_path, boxes) == ["A", "B", "C"]


def test_two_column_page_is_read_column_by_column(tmp_path):
    boxes = [(60, 50, 552, para("TITLE"))]  # spans both columns
    for i, y in enumerate([120, 220, 320], start=1):
        boxes.append((60, y, 296, para(f"L{i}")))        # left column
        boxes.append((316, y + 52, 552, para(f"R{i}")))  # right column, a little lower
    assert reading_order(tmp_path, boxes) == ["TITLE", "L1", "L2", "L3", "R1", "R2", "R3"]


def test_wide_caption_splits_the_columns_into_bands(tmp_path):
    # Boxes are drawn column by column, the way LaTeX writes a PDF.
    boxes = [
        (60, 80, 296, para("L1")), (60, 150, 296, para("L2")),     # left column, above
        (316, 120, 552, para("R1")), (316, 190, 552, para("R2")),  # right column, above
        (60, 280, 552, para("FIGURE")),                            # a caption across both
        (60, 360, 296, para("L3")), (316, 400, 552, para("R3")),   # below the caption
    ]
    assert reading_order(tmp_path, boxes) == ["L1", "L2", "R1", "R2", "FIGURE", "L3", "R3"]


# --------------------------------------------------------------------- packing
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


def test_overlap_shrinks_so_a_long_paragraph_still_fits():
    blocks = [("first " * 80).strip(), ("second " * 120).strip()]  # 479 and 839 characters
    first, second = _pack(blocks)
    assert len(second) <= CHUNK_CHARS
    carried = len(second) - len(blocks[1]) - 1  # the overlap at the start of `second`
    assert 0 < carried < CHUNK_OVERLAP
    assert second[:carried] == first[-carried:]


def test_zero_overlap_repeats_nothing(monkeypatch):
    monkeypatch.setattr(chunking, "CHUNK_OVERLAP", 0)
    blocks = [(f"Paragraph {i} " + "word " * 60).strip() for i in range(6)]
    chunks = list(_pack(blocks))
    assert all(len(c) <= CHUNK_CHARS for c in chunks)
    assert " ".join(chunks) == " ".join(blocks)  # every block exactly once


# ------------------------------------------------------------------ whole PDFs
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
