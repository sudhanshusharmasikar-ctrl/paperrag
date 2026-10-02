"""Building, saving and loading the FAISS index and its chunk file."""
import json

import pytest

from app import index as index_mod


def test_build_writes_an_index_and_chunk_file_that_match(built_index):
    assert built_index["pdfs"] == 2
    assert built_index["dim"] == 384
    index, chunks = index_mod.load()
    assert index.ntotal == len(chunks) == built_index["chunks"]
    assert {"chunk_id", "doc_id", "source", "page", "text"} <= set(chunks[0])


def test_load_says_how_to_build_a_missing_index(missing_index):
    with pytest.raises(FileNotFoundError, match="python -m app.index"):
        index_mod.load()


def test_load_refuses_an_index_out_of_sync_with_its_chunks(built_index):
    with open(index_mod.CHUNKS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_id": "extra", "text": "not in the index"}) + "\n")
    with pytest.raises(RuntimeError, match="out of sync"):
        index_mod.load()


def test_build_stops_when_there_are_no_pdfs(tmp_path):
    with pytest.raises(SystemExit, match="No PDFs found"):
        index_mod.build(tmp_path)
