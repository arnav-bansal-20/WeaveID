#!/usr/bin/env python3
"""Download specified Hugging Face datasets, record metadata, and inspect Flipkart data."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import DatasetCard, HfApi, snapshot_download

DATASETS = [
    "cansa/Describable-Textures-Dataset-DTD",
    "acseckin/FabricNET",
    "shrimantasatpati/Saree-NIFT-Style",
    "TheFatBallerina/flipkart-data-indian",
]

FIVE_GB_BYTES = 5 * 1024 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASETS_MD = PROJECT_ROOT / "data" / "DATASETS.md"


def format_bytes(num_bytes: int | None) -> str:
    """Format byte count into human-readable units."""
    if num_bytes is None:
        return "unknown"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def is_already_downloaded(dest_dir: Path) -> bool:
    """Return True if dest_dir exists and contains at least one non-cache file."""
    if not dest_dir.exists() or not dest_dir.is_dir():
        return False
    for path in dest_dir.rglob("*"):
        if path.is_file() and ".cache" not in path.parts:
            return True
    return False


def dir_stats(dest_dir: Path) -> tuple[int, int]:
    """Return (file_count, total_bytes) excluding huggingface .cache files."""
    if not dest_dir.exists():
        return 0, 0
    file_count = 0
    total_bytes = 0
    for path in dest_dir.rglob("*"):
        if path.is_file() and ".cache" not in path.parts:
            file_count += 1
            total_bytes += path.stat().st_size
    return file_count, total_bytes


def get_repo_size_and_info(api: HfApi, repo_id: str) -> tuple[int | None, Any | None, str | None]:
    """Fetch repo size from Hugging Face Hub without downloading files."""
    try:
        info = api.dataset_info(repo_id=repo_id, files_metadata=True)
        total = 0
        if info.siblings:
            for sibling in info.siblings:
                if sibling.size is not None:
                    total += sibling.size
        return total, info, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def extract_license(repo_id: str, dest_dir: Path, info: Any | None, token: str | None) -> str:
    """Read dataset card (local README.md or Hub) and return license string."""
    readme_path = dest_dir / "README.md"
    card = None
    if readme_path.exists():
        try:
            card = DatasetCard.load(readme_path)
        except Exception:
            card = None

    if card is None:
        try:
            card = DatasetCard.load(repo_id, token=token)
        except Exception:
            card = None

    if card is not None and getattr(card, "data", None) is not None:
        lic = getattr(card.data, "license", None)
        if lic:
            if isinstance(lic, list):
                return ", ".join(str(x) for x in lic)
            return str(lic)

    if info is not None and getattr(info, "card_data", None) is not None:
        lic = getattr(info.card_data, "license", None)
        if lic:
            if isinstance(lic, list):
                return ", ".join(str(x) for x in lic)
            return str(lic)

    return "Not specified"


def extract_row_count(dest_dir: Path, info: Any | None) -> str:
    """Try to determine row count from card metadata or local dataset files."""
    if info is not None and getattr(info, "card_data", None) is not None:
        ds_info = getattr(info.card_data, "dataset_info", None)
        if isinstance(ds_info, dict) and "splits" in ds_info:
            total = sum(s.get("num_examples", 0) for s in ds_info["splits"] if isinstance(s, dict))
            if total > 0:
                return str(total)
        elif isinstance(ds_info, list):
            total = 0
            for item in ds_info:
                if isinstance(item, dict) and "splits" in item:
                    total += sum(s.get("num_examples", 0) for s in item["splits"] if isinstance(s, dict))
            if total > 0:
                return str(total)
    return "N/A"


def is_auth_error(err_msg: str) -> bool:
    """Check if an error message indicates missing/invalid HF_TOKEN or gated access."""
    lower = err_msg.lower()
    return any(
        kw in lower
        for kw in ("401", "403", "gated", "unauthorized", "forbidden", "authentication", "token")
    )


def inspect_flipkart_dataset(dest_dir: Path) -> None:
    """Inspect Flipkart dataset for saree or ethnic-wear products with prices."""
    print("\n--- Inspecting Flipkart Dataset ---")
    if not is_already_downloaded(dest_dir):
        print("Flipkart dataset is not downloaded; skipping inspection.")
        return

    try:
        ds_dict = load_dataset(str(dest_dir))
    except Exception as exc:
        print(f"Could not load Flipkart dataset with `datasets`: {type(exc).__name__}: {exc}")
        return

    all_rows: list[dict[str, Any]] = []
    columns: list[str] = []
    for split_name, ds in ds_dict.items():
        print(f"Split '{split_name}': {len(ds)} rows, columns: {ds.column_names}")
        if not columns:
            columns = list(ds.column_names)
        for row in ds:
            all_rows.append(row)

    if not all_rows:
        print("Flipkart dataset has no rows.")
        return

    # Identify candidate columns
    lower_cols = {c.lower(): c for c in columns}

    def find_col(patterns: list[str]) -> str | None:
        for pat in patterns:
            for lc, orig in lower_cols.items():
                if pat in lc:
                    return orig
        return None

    name_col = find_col(["product_name", "title", "name", "product"])
    cat_col = find_col(["category", "sub_category", "subcategory", "product_category_tree", "cat"])
    price_col = find_col(["selling_price", "discounted_price", "retail_price", "price", "mrp"])

    print(f"Detected columns -> name: {name_col}, category: {cat_col}, price: {price_col}")

    if not name_col or not price_col:
        print("Flipkart dataset does not have both product name and price columns. Not creating CSV.")
        return

    keywords = re.compile(
        r"\b(saree|sarees|sari|saris|lehenga|kurta|kurti|salwar|kameez|dupatta|ethnic|anarkali|choli|dhoti|sherwani)\b",
        re.IGNORECASE,
    )

    matched_rows: list[dict[str, str]] = []
    for row in all_rows:
        name_val = str(row.get(name_col) or "").strip()
        cat_val = str(row.get(cat_col) or "").strip() if cat_col else ""
        price_val = row.get(price_col)
        text_to_search = f"{name_val} {cat_val}"

        if keywords.search(text_to_search) and price_val is not None and str(price_val).strip() not in ("", "None", "nan"):
            matched_rows.append(
                {
                    "product_name": name_val,
                    "category": cat_val,
                    "price": str(price_val).strip(),
                }
            )

    if not matched_rows:
        print("No saree or ethnic-wear products with prices found in the Flipkart dataset. CSV not created.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROCESSED_DIR / "saree_prices.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product_name", "category", "price"])
        writer.writeheader()
        writer.writerows(matched_rows)

    print(f"Found {len(matched_rows)} saree/ethnic-wear products with prices.")
    print(f"Saved filtered CSV to {out_csv}")


def write_datasets_md(records: list[dict[str, str]]) -> None:
    """Write metadata and licenses to data/DATASETS.md."""
    DATASETS_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Datasets",
        "",
        "| Repo ID | Status | License | Files | Rows | Size on Disk | Download Date | Local Path |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        lines.append(
            f"| `{r['repo_id']}` | {r['status']} | {r['license']} | {r['files']} | "
            f"{r['rows']} | {r['size']} | {r['date']} | `{r['path']}` |"
        )
    lines.append("")
    DATASETS_MD.write_text("\n".join(lines), encoding="utf-8")


def print_summary_table(records: list[dict[str, str]]) -> None:
    """Print summary table: dataset, status, size, license, folder path."""
    print("\n=== Download Summary ===")
    headers = ("Dataset", "Status", "Size", "License", "Folder Path")
    rows = [
        (r["repo_id"], r["status"], r["size"], r["license"], r["path"])
        for r in records
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Hugging Face datasets.")
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Confirm download without prompting if > 5 GB."
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=hf_token)

    print("Checking repository sizes on Hugging Face Hub...")
    repo_checks: dict[str, tuple[int | None, Any | None, str | None]] = {}
    total_remote_bytes = 0

    for repo_id in DATASETS:
        size_bytes, info, err = get_repo_size_and_info(api, repo_id)
        repo_checks[repo_id] = (size_bytes, info, err)
        if err:
            print(f"  - {repo_id}: could not fetch size ({err})")
            if is_auth_error(err) and not hf_token:
                print("    -> Note: HF_TOKEN environment variable is not set and may be required.")
        else:
            total_remote_bytes += size_bytes or 0
            print(f"  - {repo_id}: {format_bytes(size_bytes)}")

    print(f"Combined known size: {format_bytes(total_remote_bytes)}")
    if total_remote_bytes > FIVE_GB_BYTES and not args.yes:
        try:
            reply = input("Combined size exceeds 5 GB. Proceed with download? [y/N]: ").strip().lower()
        except EOFError:
            print("Combined size exceeds 5 GB and no interactive input is available. Re-run with --yes to confirm.")
            return 2
        if reply not in ("y", "yes"):
            print("Download cancelled by user.")
            return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    records: list[dict[str, str]] = []
    token_needed_repos: list[str] = []

    for repo_id in DATASETS:
        dataset_name = repo_id.split("/")[-1]
        dest_dir = RAW_DIR / dataset_name
        rel_path = str(dest_dir.relative_to(PROJECT_ROOT))
        _, info, check_err = repo_checks[repo_id]

        print(f"\nProcessing {repo_id} -> {rel_path}/")
        status = "ok"
        error_msg = None

        if is_already_downloaded(dest_dir):
            print(f"  Already downloaded at {rel_path}/. Skipping download.")
        else:
            try:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=str(dest_dir),
                    token=hf_token,
                )
                print(f"  Downloaded successfully to {rel_path}/")
            except Exception as exc:
                status = "failed"
                error_msg = f"{type(exc).__name__}: {exc}"
                print(f"  ERROR downloading {repo_id}: {error_msg}", file=sys.stderr)
                if is_auth_error(error_msg):
                    token_needed_repos.append(repo_id)

        if status == "ok":
            file_count, disk_bytes = dir_stats(dest_dir)
            lic = extract_license(repo_id, dest_dir, info, hf_token)
            rows_str = extract_row_count(dest_dir, info)
            size_str = format_bytes(disk_bytes)
            files_str = str(file_count)
        else:
            lic = extract_license(repo_id, dest_dir, info, hf_token) if info else "unknown"
            rows_str = "N/A"
            size_str = "0 B"
            files_str = "0"

        records.append(
            {
                "repo_id": repo_id,
                "status": status,
                "license": lic,
                "files": files_str,
                "rows": rows_str,
                "size": size_str,
                "date": today,
                "path": rel_path,
                "error": error_msg or "",
            }
        )

    write_datasets_md(records)
    print(f"\nWrote dataset metadata to {DATASETS_MD.relative_to(PROJECT_ROOT)}")

    flipkart_dir = RAW_DIR / "flipkart-data-indian"
    inspect_flipkart_dataset(flipkart_dir)

    if token_needed_repos and not hf_token:
        print(
            "\nNOTE: One or more datasets failed due to authentication/gating "
            f"({', '.join(token_needed_repos)}). Set the HF_TOKEN environment variable and re-run."
        )

    print_summary_table(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
