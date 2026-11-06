"""Build the VLNVerse challenge manifest that the backend reads episodes from.

The VLNVerse challenge scores 150 published trajectories per track on the
official server. This script selects exactly those rows from the official test
split, in published order, without sampling.

    python -m cua_embodied.apps.vlnverse.build_manifest --granularity coarse \\
        --test data/vlnverse/coarse_test.json.gz \\
        --challenge-subset data/vlnverse/challenge_subset.txt \\
        --out data/vlnverse/coarse_challenge150_manifest.json

Sources: coarse_test.json.gz and fine_test.json.gz from the Eyz/VLNVerse_data
dataset (raw_data/final_splits), and scripts/challenge_subset.txt from
https://github.com/sihaoevery/vlnverse_emr.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def challenge_rows(path: Path, granularity: str) -> list[dict[str, str]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"malformed challenge row {number}: {line!r}")
        row_granularity, trajectory_id, episode_id, scene_id = fields
        if row_granularity == granularity:
            rows.append(dict(trajectory_id=trajectory_id, episode_id=episode_id, scene_id=scene_id))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--test", type=Path, required=True, help="coarse_test.json.gz or fine_test.json.gz")
    parser.add_argument("--granularity", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--challenge-subset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with gzip.open(args.test, "rt", encoding="utf-8") as stream:
        test_episodes = json.load(stream)["episodes"]
    by_id = {e["trajectory_id"]: (i, e) for i, e in enumerate(test_episodes)}
    rows = challenge_rows(args.challenge_subset, args.granularity)
    if len(rows) != 150 or len({r["trajectory_id"] for r in rows}) != 150:
        raise ValueError(f"expected 150 distinct {args.granularity} challenge rows, got {len(rows)}")

    episodes = []
    for challenge_index, row in enumerate(rows):
        source_index, source = by_id[row["trajectory_id"]]
        if source["episode_id"] != row["episode_id"] or source["scene_id"] != row["scene_id"]:
            raise ValueError(f"challenge row does not match the test split: {row}")
        text = source["instruction"]["instruction_text"]
        # The official coarse evaluation uses the formal instruction. Fine
        # annotations have one step-by-step instruction per trajectory.
        instruction = text["formal"] if args.granularity == "coarse" else text
        episodes.append(dict(
            challenge_index=challenge_index, source_test_index=source_index,
            episode_id=source["episode_id"], trajectory_id=source["trajectory_id"],
            scan=source["scan"], scene_id=source["scene_id"],
            start_position=source["start_position"], start_rotation=source["start_rotation"],
            instruction=instruction))

    manifest = dict(benchmark="VLNVerse", granularity=args.granularity, split="challenge",
                    episodes=episodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(episodes)} {args.granularity} episodes to {args.out}")


if __name__ == "__main__":
    main()
