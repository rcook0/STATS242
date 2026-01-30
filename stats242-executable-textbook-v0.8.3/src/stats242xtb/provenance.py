from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict

def _sha256_file(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _safe_version(pkg: str) -> str | None:
    try:
        import importlib.metadata as md
        return md.version(pkg)
    except Exception:
        return None

def collect_env() -> Dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {
            "numpy": _safe_version("numpy"),
            "pandas": _safe_version("pandas"),
            "scipy": _safe_version("scipy"),
            "statsmodels": _safe_version("statsmodels"),
            "matplotlib": _safe_version("matplotlib"),
            "pyyaml": _safe_version("pyyaml"),
            "markdown": _safe_version("markdown"),
        },
    }

def maybe_git_hash(repo_root: str) -> str | None:
    head = os.path.join(repo_root, ".git", "HEAD")
    if not os.path.exists(head):
        return None
    try:
        with open(head, "r", encoding="utf-8") as f:
            ref = f.read().strip()
        if ref.startswith("ref:"):
            ref_path = ref.split(":", 1)[1].strip()
            p = os.path.join(repo_root, ".git", ref_path)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return f.read().strip()
        return ref
    except Exception:
        return None

def write_run_meta(
    *,
    out_dir: str,
    chapter: str,
    params: Dict[str, Any],
    inputs: Dict[str, str] | None = None,
    extra: Dict[str, Any] | None = None,
    repo_root: str | None = None,
) -> str:
    os.makedirs(os.path.join(out_dir, "artifacts"), exist_ok=True)
    inputs = inputs or {}
    inp_meta: Dict[str, Any] = {}
    for k, p in inputs.items():
        if p is None:
            continue
        try:
            inp_meta[k] = {
                "path": str(p),
                "exists": bool(os.path.exists(str(p))),
                "sha256": _sha256_file(str(p)) if os.path.exists(str(p)) else None,
            }
        except Exception as e:
            inp_meta[k] = {"path": str(p), "error": str(e)}

    root = repo_root or os.getcwd()
    meta: Dict[str, Any] = {
        "chapter": chapter,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "params": params,
        "inputs": inp_meta,
        "env": collect_env(),
        "git": {"hash": maybe_git_hash(root)},
    }
    if extra:
        meta["extra"] = extra

    out_path = os.path.join(out_dir, "artifacts", "run_meta.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return out_path
