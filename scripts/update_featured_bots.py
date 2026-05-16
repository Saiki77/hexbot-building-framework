#!/usr/bin/env python3
"""Regenerate the Featured Bots table in README.md from leaderboard.json.

Reads `leaderboard.json` (entries are written by `orca.leaderboard.Leaderboard.rate`),
filters out reference bots, and rewrites the markdown block between
`<!-- BOTS:START -->` and `<!-- BOTS:END -->` markers in README.md.

Designed to run as a GitHub Action; safe to run locally too.

Usage:
    python scripts/update_featured_bots.py
    python scripts/update_featured_bots.py --top 20
    python scripts/update_featured_bots.py --leaderboard path/to/leaderboard.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

# Reference bots ship with the framework; exclude them from the community gallery.
REFERENCE_NAMES = {"random", "heuristic", "orca"}

START_MARKER = "<!-- BOTS:START -->"
END_MARKER = "<!-- BOTS:END -->"


def load_entries(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def build_table(entries: list, top_n: int) -> str:
    community = [
        e for e in entries
        if e.get("name", "").lower() not in REFERENCE_NAMES
    ]
    community.sort(key=lambda e: e.get("elo", 0), reverse=True)
    community = community[:top_n]

    if not community:
        return ("_No community bots rated yet. "
                "Train one, then `Leaderboard().rate(bot, name='my-bot-v1')` "
                "to land here. See "
                "[Evaluation and Sharing]"
                "(https://github.com/Saiki77/hexbot-building-framework/wiki/Evaluation-and-Sharing)._")

    lines = [
        "| Rank | Bot | ELO | Win rate | Games | Rated |",
        "|---|---|---:|---:|---:|---|",
    ]
    for rank, e in enumerate(community, start=1):
        name = e.get("name", "?")
        elo = int(round(e.get("elo", 0)))
        win_rate = e.get("overall_win_rate", 0)
        games = e.get("total_games", 0)
        rated_at = e.get("rated_at", 0)
        rated_str = (dt.datetime.fromtimestamp(rated_at).date().isoformat()
                     if rated_at else "?")
        lines.append(
            f"| {rank} | {name} | {elo} | {win_rate*100:.0f}% | "
            f"{games} | {rated_str} |"
        )
    return "\n".join(lines)


def patch_readme(readme_path: str, table: str) -> bool:
    """Replace content between markers; return True if README changed."""
    with open(readme_path) as f:
        content = f.read()

    if START_MARKER not in content or END_MARKER not in content:
        # Markers not present yet; insert a stub Featured Bots section
        # near the top, after the first level-2 heading (## Installation)
        # or fall through to no-op if we can't find a stable anchor.
        return False

    pattern = re.compile(
        re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
        re.DOTALL,
    )
    replacement = f"{START_MARKER}\n{table}\n{END_MARKER}"
    new_content = pattern.sub(replacement, content)

    if new_content == content:
        return False
    with open(readme_path, "w") as f:
        f.write(new_content)
    return True


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leaderboard", default="leaderboard.json",
        help="Path to leaderboard.json (default: ./leaderboard.json)",
    )
    parser.add_argument(
        "--readme", default="README.md",
        help="Path to README.md to update (default: ./README.md)",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of bots to feature (default: 10)",
    )
    args = parser.parse_args(argv)

    entries = load_entries(args.leaderboard)
    table = build_table(entries, args.top)
    changed = patch_readme(args.readme, table)
    if changed:
        print(f"Updated {args.readme} with {min(args.top, len(entries))} community bots")
    else:
        print(f"No change to {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
