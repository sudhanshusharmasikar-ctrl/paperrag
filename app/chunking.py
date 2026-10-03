"""
Layout-aware chunking.

The naive approach is: extract the whole PDF as one string, split every N chars.
That destroys page boundaries, so you can never cite a page, and it happily
splices the end of a two-column page into the start of the next one.

This module instead walks PyMuPDF text *blocks*. A block is roughly a visual
paragraph. PyMuPDF's own block order doesn't follow columns, so _reading_order
sorts each page's blocks column by column on two-column pages. Chunks are then
assembled from whole blocks and never cross a page boundary, so every chunk
carries exactly one page number.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import pymupdf as fitz  # PyMuPDF; the bare "fitz" import name is deprecated

from .config import CHUNK_CHARS, CHUNK_OVERLAP, COLUMN_TOLERANCE, MIN_CHUNK_CHARS


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str       # filename stem, e.g. "tmpt_2024"
    source: str       # original filename, shown to the user
    page: int         # 1-indexed, matches what a human sees in a PDF reader
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


# Ligatures and hyphen-wrapping are the two things that most corrupt academic
# PDF text and silently degrade embedding quality.
_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\u2019": "'", "\u201c": '"',
    "\u201d": '"', "\u2013": "-", "\u2014": "-",
}


def clean(text: str) -> str:
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    # join words broken across a line by a hyphen: "multi-\nmodal" -> "multimodal"
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _is_noise(block_text: str) -> bool:
    """Drop running headers, bare page numbers, and figure/table stubs."""
    t = block_text.strip()
    if len(t) < 25:
        return True
    if re.fullmatch(r"[\d\s\-–.ivxlIVXL]+", t):
        return True
    # a block that is mostly digits and punctuation is almost always a table row
    alpha = sum(c.isalpha() for c in t)
    return alpha / max(len(t), 1) < 0.45


def _reading_order(blocks: list, mid: float) -> list:
    """
    Sort one page's blocks into reading order, for one- and two-column pages.

    A block that crosses the middle of the page (title, abstract, a wide figure
    caption) splits the page into horizontal bands. Inside each band the left
    column is read top to bottom, then the right column. On a one-column page
    almost every block crosses the middle, so this is plain top-to-bottom.
    """
    def side(b) -> int:  # -1 left column, 1 right column, 0 crosses the middle
        x0, x1 = b[0], b[2]
        if x1 <= mid + COLUMN_TOLERANCE:
            return -1
        if x0 >= mid - COLUMN_TOLERANCE:
            return 1
        return 0

    def top_left(b):
        return (round(b[1], 1), round(b[0], 1))

    wide = sorted((b for b in blocks if side(b) == 0), key=top_left)
    tops = [b[1] for b in wide]
    bands: list[list] = [[] for _ in range(len(wide) + 1)]
    for b in blocks:
        if side(b) != 0:
            bands[bisect_right(tops, b[1])].append(b)  # band = wide blocks above it

    ordered = []
    for i, band in enumerate(bands):
        if i > 0:
            ordered.append(wide[i - 1])
        ordered += sorted((b for b in band if side(b) < 0), key=top_left)
        ordered += sorted((b for b in band if side(b) > 0), key=top_left)
    return ordered


def page_blocks(page: "fitz.Page") -> list[str]:
    """Text blocks on one page, in reading order, cleaned and de-noised."""
    raw = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    text_blocks = [b for b in raw if len(b) > 6 and b[6] == 0]
    mid = (page.rect.x0 + page.rect.x1) / 2
    out = []
    for b in _reading_order(text_blocks, mid):
        t = clean(b[4])
        if t and not _is_noise(t):
            out.append(t)
    return out


def _overlap(chunk: str, next_len: int) -> str:
    """
    The end of `chunk` to repeat at the start of the next chunk: CHUNK_OVERLAP
    characters, or fewer when the next piece is long, so the new chunk still
    fits in CHUNK_CHARS.
    """
    n = min(CHUNK_OVERLAP, CHUNK_CHARS - next_len - 1)  # -1 for the joining space
    # not chunk[-n:], because chunk[-0:] is the whole chunk
    return chunk[len(chunk) - n:] if n > 0 else ""


def _pack(blocks: list[str]) -> Iterator[str]:
    """
    Greedily pack whole blocks into chunks of at most CHUNK_CHARS, carrying the
    end of the previous chunk forward (see _overlap) so a sentence split across
    a chunk edge is still retrievable from at least one side. Only a single
    sentence longer than CHUNK_CHARS can make a longer chunk.

    `size` and `cur_len` count every piece plus the space that follows it, i.e.
    the joined chunk's length + 1.
    """
    buf: list[str] = []
    size = 0
    for block in blocks:
        # a single oversized block (dense related-work paragraph) is split on
        # sentence boundaries rather than mid-word
        if len(block) > CHUNK_CHARS:
            if buf:
                yield " ".join(buf)
                buf, size = [], 0
            sentences = re.split(r"(?<=[.!?])\s+", block)
            cur: list[str] = []
            cur_len = 0
            for s in sentences:
                if cur_len + len(s) > CHUNK_CHARS and cur:
                    chunk = " ".join(cur)
                    yield chunk
                    tail = _overlap(chunk, len(s))
                    cur = [tail, s] if tail else [s]
                    cur_len = len(" ".join(cur)) + 1
                else:
                    cur.append(s)
                    cur_len += len(s) + 1
            if cur:
                yield " ".join(cur)
            continue

        if size + len(block) > CHUNK_CHARS and buf:
            chunk = " ".join(buf)
            yield chunk
            tail = _overlap(chunk, len(block))
            buf = [tail, block] if tail else [block]
            size = len(" ".join(buf)) + 1
        else:
            buf.append(block)
            size += len(block) + 1

    if buf:
        yield " ".join(buf)


def chunk_pdf(path: Path) -> list[Chunk]:
    doc_id = path.stem
    chunks: list[Chunk] = []
    with fitz.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            blocks = page_blocks(page)
            if not blocks:
                continue  # scanned page with no text layer; see README on OCR
            for i, text in enumerate(_pack(blocks)):
                if len(text) < MIN_CHUNK_CHARS:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::p{page_no}::c{i}",
                        doc_id=doc_id,
                        source=path.name,
                        page=page_no,
                        text=text,
                    )
                )
    return chunks
