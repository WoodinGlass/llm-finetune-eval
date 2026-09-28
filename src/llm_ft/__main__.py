"""CLI entrypoint: `python -m llm_ft [info|version|help]`.

Keep it tiny — this is a diagnostic tool, not the serving entrypoint
(serving lives in `serving/` and comes online at M5).
"""

from __future__ import annotations

import json
import sys

from llm_ft import __version__
from llm_ft.env import info as env_info


def _usage() -> str:
    return (
        "usage: python -m llm_ft <command>\n\n"
        "commands:\n"
        "  info      print detected platform and standard paths (JSON)\n"
        "  version   print package version\n"
        "  help      show this message\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = (argv if argv is not None else sys.argv[1:]) or ["info"]
    cmd = args[0]

    if cmd == "version":
        print(__version__)
        return 0
    if cmd == "info":
        print(json.dumps(env_info(), indent=2, default=str))
        return 0
    if cmd in {"help", "-h", "--help"}:
        print(_usage(), end="")
        return 0

    print(f"unknown command: {cmd}\n", file=sys.stderr)
    print(_usage(), file=sys.stderr, end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
