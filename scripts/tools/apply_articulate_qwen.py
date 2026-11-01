#!/usr/bin/env python3
"""Teach deps/articulate-anything to use Qwen3-VL (Bailian, OpenAI-compatible API).

articulate-anything builds its own VLM clients (``articulate_anything/utils/vlm.py``), so the
``SIMFOUNDRY_VLM_BACKEND`` switch in ``simfoundry/models/vlm.py`` never reaches its subprocess.
This adds a ``qwen`` branch to the two places that pick a client by model name, plus one new file.

deps/ is not tracked by this repo, so the result is meant to be turned into
``patches/articulate-anything-qwen.patch`` (see the end of the run) rather than kept as an edit.

Each edit is anchored on exact text and refuses to run if the anchor is missing or ambiguous, so
an upstream change fails loudly instead of corrupting a file. Running it twice is a no-op.

Run on the machine that has the checkout:

    python scripts/tools/apply_articulate_qwen.py
    python scripts/tools/apply_articulate_qwen.py --repo /data/SimFoundry/deps/articulate-anything
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW_FILE_SOURCE = HERE / "articulate_qwen" / "qwen_vlm.py"
NEW_FILE_DEST = Path("articulate_anything/utils/qwen_vlm.py")

QUERY_VLM = Path("simfoundry/query_vlm.py")
PROMPT_UTILS = Path("articulate_anything/utils/prompt_utils.py")

EDITS = [
    (
        QUERY_VLM,
        '    elif "gpt" in cfg.model_name:\n',
        '    elif "qwen" in cfg.model_name:\n'
        "        from articulate_anything.utils.qwen_vlm import QwenVLM\n"
        "        return QwenVLM(model=cfg.model_name, verbose=verbose)\n"
        '    elif "gpt" in cfg.model_name:\n',
    ),
    (
        PROMPT_UTILS,
        "        return setup_claude(model_name, system_instruction, api_key, cfg)\n    else:\n",
        "        return setup_claude(model_name, system_instruction, api_key, cfg)\n"
        '    elif "qwen" in model_name:\n'
        "        from articulate_anything.utils.qwen_vlm import QwenWrapper\n"
        "        return QwenWrapper(model_name, system_instruction)\n"
        "    else:\n",
    ),
]


def apply_edit(repo: Path, rel: Path, anchor: str, replacement: str) -> str:
    path = repo / rel
    text = path.read_text()
    if replacement in text:
        return "already applied"
    count = text.count(anchor)
    if count != 1:
        raise SystemExit(
            f"{rel}: expected the anchor exactly once, found {count}. Upstream changed; "
            f"edit by hand.\nanchor:\n{anchor}"
        )
    path.write_text(text.replace(anchor, replacement))
    return "patched"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", default="/data/SimFoundry/deps/articulate-anything")
    args = parser.parse_args()
    repo = Path(args.repo)
    if not (repo / PROMPT_UTILS).is_file():
        raise SystemExit(f"{repo} does not look like articulate-anything ({PROMPT_UTILS} missing)")

    dest = repo / NEW_FILE_DEST
    if dest.is_file() and dest.read_text() == NEW_FILE_SOURCE.read_text():
        print(f"{NEW_FILE_DEST}: already applied")
    else:
        shutil.copyfile(NEW_FILE_SOURCE, dest)
        print(f"{NEW_FILE_DEST}: written")
    for rel, anchor, replacement in EDITS:
        print(f"{rel}: {apply_edit(repo, rel, anchor, replacement)}")

    # Other spots that pick behaviour from the model name may still treat qwen as unknown.
    print("\nModel-name checks left to review (a qwen model reaches none of these as gemini):")
    pattern = re.compile(r'"gemini" in|\'gemini\' in|startswith\("gemini')
    for source in sorted(repo.glob("**/*.py")):
        if ".git" in source.parts or source == dest:
            continue
        for number, line in enumerate(source.read_text().splitlines(), 1):
            if pattern.search(line):
                print(f"  {source.relative_to(repo)}:{number}: {line.strip()}")

    print(
        "\nTo record this as a patch (run in the checkout):\n"
        f"  git -C {repo} add -N {NEW_FILE_DEST}\n"
        f"  git -C {repo} diff > /data/SimFoundry/patches/articulate-anything-qwen.patch"
    )


if __name__ == "__main__":
    sys.exit(main())
