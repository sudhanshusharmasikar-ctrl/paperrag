"""Settings come from a .env file in the project folder, and the shell wins over it."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "app" / "config.py"


def test_dotenv_is_read_and_a_shell_variable_overrides_it(tmp_path):
    # A copy of app/config.py in an empty folder, so the real project's .env
    # (if you have one) can't affect the result.
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    shutil.copy(CONFIG, tmp_path / "app" / "config.py")
    (tmp_path / ".env").write_text("PAPERRAG_TOP_K=7\nPAPERRAG_GEN_MODE=mistral\n")

    env = {k: v for k, v in os.environ.items() if not k.startswith("PAPERRAG_")}
    env["PAPERRAG_GEN_MODE"] = "extractive"  # set in the shell, so it beats the file
    out = subprocess.run(
        [sys.executable, "-c", "from app import config; print(config.TOP_K, config.GEN_MODE)"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    assert out.stdout.split() == ["7", "extractive"]
