import json
from pathlib import Path


def read_jsonl(filepath: str) -> list[list]:
    """Read JSONL file where each line is a JSON array."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(filepath: str, rows: list[list]) -> None:
    """Write rows as JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def combine_jsonl_files(
    base_file: str,
    augment_file: str,
    output_file: str,
    record_id_index: int = 0
) -> None:
    """
    Combine two JSONL files, re-indexing record_id in the augmented file.
    
    Args:
        base_file: Path to the base JSONL file
        augment_file: Path to the file to append
        output_file: Path for the combined output
        record_id_index: Index of record_id in each row (default: 0)
    """
    # Read base file
    base_rows = read_jsonl(base_file)
    
    # Skip header row if present (check if first element is "record_id")
    has_header = False
    if base_rows and base_rows[0][record_id_index] == "record_id":
        header = base_rows[0]
        base_rows = base_rows[1:]
        has_header = True
    
    # Find max record_id in base file
    max_record_id = 0
    for row in base_rows:
        rid = row[record_id_index]
        if isinstance(rid, int) and rid > max_record_id:
            max_record_id = rid
    
    print(f"📊 Base file: {len(base_rows)} rows, max record_id: {max_record_id}")
    
    # Read augment file
    augment_rows = read_jsonl(augment_file)
    
    # Skip header in augment file if present
    if augment_rows and augment_rows[0][record_id_index] == "record_id":
        augment_rows = augment_rows[1:]
    
    print(f"📊 Augment file: {len(augment_rows)} rows")
    
    # Re-index augmented rows
    for row in augment_rows:
        max_record_id += 1
        row[record_id_index] = max_record_id
    
    # Combine all rows
    all_rows = []
    if has_header:
        all_rows.append(header)
    all_rows.extend(base_rows)
    all_rows.extend(augment_rows)
    
    # Write output
    write_jsonl(output_file, all_rows)
    
    print(f"✅ Combined {len(base_rows)} + {len(augment_rows)} = {len(base_rows) + len(augment_rows)} rows")
    print(f"💾 Saved to {output_file}")


if __name__ == "__main__":
    combine_jsonl_files(
        base_file="answers/R2_train.jsonl",
        augment_file="answers/sutd_train_gt_generated.jsonl",
        output_file="answers/R2_train_augmented.jsonl"
    )