"""Concatenate dfs files in time."""
# /// script
# requires-python = ">=3.12"
# dependencies = ["mikeio==3.3.0"]
# [tool.uv]
# exclude-newer = "2026-09-23T00:00:00Z"
# ///

import argparse
import glob
from collections.abc import Sequence

from mikeio.generic import concat


def main(argv: Sequence[str] | None = None) -> None:
    """Concatenate dfs files in time."""
    parser = argparse.ArgumentParser(description="Concatenate dfs files in time.")
    parser.add_argument("inputs", nargs="+", help="input files or glob patterns")
    parser.add_argument("output", help="output file")
    args = parser.parse_args(argv)

    # Windows shells pass wildcards through unexpanded.
    infiles = sorted(f for pattern in args.inputs for f in glob.glob(pattern))
    if not infiles:
        parser.error(f"no files match {args.inputs}")

    concat(infilenames=infiles, outfilename=args.output)
    print(f"Created {args.output} from {len(infiles)} files")


if __name__ == "__main__":
    main()
