"""Settings come from a .env file in the project folder, and the shell wins over it."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "app" / "config.py"


def read_config(tmp_path, expr, dotenv=None, **shell):
    """Evaluate `expr` on a copy of app/config.py in an empty folder, so the real
    project's .env (if you have one) and your shell can't affect the result."""
    (tmp_path / "app").mkdir(parents=True)
    (tmp_path / "app" / "__init__.py").write_text("")
    shutil.copy(CONFIG, tmp_path / "app" / "config.py")
    if dotenv is not None:
        (tmp_path / ".env").write_text(dotenv)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PAPERRAG_")}
    env.update(shell)
    out = subprocess.run(
        [sys.executable, "-c", f"from app import config; print({expr})"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    return out.stdout.split()


def test_dotenv_is_read_and_a_shell_variable_overrides_it(tmp_path):
    out = read_config(tmp_path, "config.TOP_K, config.GEN_MODE",
                      dotenv="PAPERRAG_TOP_K=7\nPAPERRAG_GEN_MODE=mistral\n",
                      PAPERRAG_GEN_MODE="extractive")  # set in the shell, so it beats the file
    assert out == ["7", "extractive"]


def test_the_library_picks_the_device_unless_one_is_set(tmp_path):
    assert read_config(tmp_path / "a", "config.EMBED_DEVICE") == ["None"]
    assert read_config(tmp_path / "b", "config.EMBED_DEVICE", PAPERRAG_DEVICE="cuda") == ["cuda"]


def test_hybrid_search_is_the_default(tmp_path):
    assert read_config(tmp_path / "a", "config.RETRIEVAL_MODE") == ["hybrid"]
    assert read_config(tmp_path / "b", "config.RETRIEVAL_MODE", PAPERRAG_RETRIEVAL="dense") == ["dense"]


def test_one_chunk_per_page_is_on_unless_turned_off(tmp_path):
    assert read_config(tmp_path / "a", "config.DISTINCT_PAGES") == ["True"]
    assert read_config(tmp_path / "b", "config.DISTINCT_PAGES", PAPERRAG_DISTINCT_PAGES="0") == ["False"]
