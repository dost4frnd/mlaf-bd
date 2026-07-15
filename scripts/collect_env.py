#!/usr/bin/env python3
"""Capture the runtime environment for reproducibility -> results/env.json."""
from __future__ import annotations
import json, platform, subprocess, sys
from pathlib import Path


def _sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def main():
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "git_commit": _sh("git rev-parse HEAD"), "pip_freeze": _sh("pip freeze").splitlines()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        env["torch"] = None
    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/env.json", "w") as fh:
        json.dump(env, fh, indent=2)
    print("wrote results/env.json")


if __name__ == "__main__":
    main()
