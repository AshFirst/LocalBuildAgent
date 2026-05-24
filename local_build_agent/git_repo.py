from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse


def is_github_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.netloc.lower() in {"github.com", "gist.github.com"}


@contextmanager
def cloned_repo(url: str, keep_clone: bool = False):
    if not is_github_url(url):
        path = Path(url)
        if path.exists() and path.is_dir():
            yield path.resolve()
            return
        raise ValueError("Input must be a GitHub/Gist HTTPS URL or an existing local repo path.")

    temp_root = Path(tempfile.mkdtemp(prefix="local-build-agent-"))
    repo_dir = temp_root / "repo"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(repo_dir)],
            check=True,
            text=True,
            capture_output=True,
        )
        yield repo_dir
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"git clone failed: {message}") from exc
    finally:
        if keep_clone:
            print(f"Kept clone at: {repo_dir}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
