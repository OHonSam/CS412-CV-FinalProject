"""
Convert a JSONL file (with a JSON-array header on line 1) to a CSV
containing only:
  - filename  (column 3 / key 'vid_filename')
  - answer    (final column / key 'answer')

Usage:
  python jsonl_to_csv.py input_path output_path
"""

import argparse
import csv
import json
from pathlib import Path


def extract_filename_answer(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8", newline="") as f_out:

        writer = csv.writer(f_out)
        writer.writerow(["id", "filename", "answer"])

        # ---- read & parse header (line 1) ----
        header_line = f_in.readline()
        if not header_line:
            raise ValueError("Input file is empty.")

        try:
            header = json.loads(header_line)
            if not isinstance(header, list):
                raise ValueError
        except Exception:
            raise ValueError(
                "First line must be a JSON array header like "
                '["record_id", ..., "answer"].'
            )

        # indices based on your header spec
        # column 3 -> index 2 (0-based)
        filename_idx = 2
        answer_idx = len(header) - 1

        # Also support dict-based lines by key
        id_key = "record_id"
        filename_key = "vid_filename"
        answer_key = "answer"

        # ---- process data lines ----
        for line_num, line in enumerate(f_in, start=2):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON on line {line_num}")

            # Case A: line is a dict
            if isinstance(obj, dict):
                if filename_key not in obj or answer_key not in obj:
                    raise ValueError(
                        f"Missing '{filename_key}' or '{answer_key}' on line {line_num}"
                    )

                record_id = obj[id_key]
                filename = obj[filename_key]
                answer = obj[answer_key]

            # Case B: line is a list/array
            elif isinstance(obj, list):
                if len(obj) <= max(filename_idx, answer_idx):
                    raise ValueError(
                        f"Line {line_num} has too few columns: {len(obj)}"
                    )

                record_id = obj[0]
                filename = obj[filename_idx]
                answer = obj[answer_idx]

            else:
                raise ValueError(
                    f"Line {line_num} must be a JSON object or array."
                )

            writer.writerow([record_id, filename, answer])

    print(f"Done. Wrote CSV to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to input .jsonl file")
    parser.add_argument("output_path", help="Path to output .csv file")
    args = parser.parse_args()

    if not args.input_path.endswith(".jsonl"):
        raise ValueError("Input file must have a .jsonl extension.")

    if not args.output_path.endswith(".csv"):
        raise ValueError("Output file must have a .csv extension.")

    extract_filename_answer(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
