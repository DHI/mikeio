""" "Concatenate DFS files."""
# /// script
# dependencies = [
#   "mikeio",
# ]
# ///

import argparse
import glob
from mikeio.generic import concat as cat


def main():
    "Concatenate DFS files."
    parser = argparse.ArgumentParser(description="Concatenate DFS files.")
    parser.add_argument(
        "files", nargs="+", help="Input DFS files followed by the output file"
    )

    args = parser.parse_args()

    if len(args.files) < 2:
        print("Error: At least one input file and one output file are required.")
        return

    output_file = args.files[-1]
    input_patterns = args.files[:-1]
    input_files = [file for pattern in input_patterns for file in glob.glob(pattern)]

    if not input_files:
        print("Error: No matching input files found.")
        return

    print("Concatenating files:", input_files)
    cat(infilenames=input_files, outfilename=output_file)
    print(f"Created: {output_file}")


if __name__ == "__main__":
    main()
