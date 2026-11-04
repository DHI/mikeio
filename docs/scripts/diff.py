"""Write the difference between two dfs files (scenario - baseline)."""
# /// script
# requires-python = ">=3.12"
# dependencies = ["mikeio==3.3.0"]
# [tool.uv]
# exclude-newer = "2026-09-23T00:00:00Z"
# ///

import argparse
from collections.abc import Sequence

from mikeio.generic import diff


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line and write the difference file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", help="scenario result file")
    parser.add_argument("baseline", help="baseline result file")
    parser.add_argument("output", help="output file")
    args = parser.parse_args(argv)

    diff(args.scenario, args.baseline, args.output)
    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
