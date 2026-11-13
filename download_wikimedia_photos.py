'''"""
Download Wikimedia Commons images listed in Sheet1 of Photo_Cropper_Evaluation.xlsx.

Important:
- Uses the FIRST occurrence of each column header. This matters because the
  workbook has a second "Primary_Category" column in its summary table.
- Does NOT deduplicate images. If the same Wikimedia URL appears in multiple
  spreadsheet rows/categories, it is downloaded separately into each category.
- Personal / Google Photos entries are skipped for manual downloading.
- Creates a CSV report with the outcome for every spreadsheet row.

Install:
    pip install openpyxl requests pillow

Run:
    python download_wikimedia_images.py
"""'''
#%%
import csv
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from tqdm import trange
import requests
from openpyxl import load_workbook
from PIL import Image


# ------------------------- CONFIGURATION -------------------------

EXCEL_FILE = Path("Photo_Cropper_Evaluation.xlsx")
SHEET_NAME = "Sheet1"
OUTPUT_DIR = Path("downloaded_images")
REPORT_NAME = "download_report.csv"

REQUEST_DELAY_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 40

HEADERS = {
    "User-Agent": (
        "PhotoCropperEvaluationDatasetDownloader/1.1 "
        "(personal research project)"
    )
}


# ------------------------- HELPER FUNCTIONS -------------------------

def safe_folder_name(value):
    """Convert a category name into a safe folder name."""
    name = str(value or "").strip()
    if not name:
        return "Uncategorised"
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", " ", name)
    return name.rstrip(". ")


def safe_id(value, fallback):
    """Format Excel IDs cleanly (e.g. 1.0 becomes 1)."""
    if value is None:
        return str(fallback)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return re.sub(r'[<>:"/\\|?*]', "_", str(value).strip())


def get_download_url(image_value):
    """
    Convert a Wikimedia Commons file-page URL to a redirect URL for the
    original file. Direct upload.wikimedia.org URLs are accepted as-is.
    """
    if not isinstance(image_value, str):
        return None, "Missing image link"

    value = image_value.strip()
    if not value.lower().startswith(("http://", "https://")):
        return None, "Not a URL (manual review)"

    parsed = urlparse(value)
    host = parsed.netloc.lower()

    # Direct Wikimedia upload URL.
    if host.endswith("wikimedia.org") and host.startswith("upload."):
        return value.split("#", 1)[0], None

    # Commons file page.
    if host in {"commons.wikimedia.org", "www.commons.wikimedia.org"}:
        path = unquote(parsed.path)

        if path.startswith("/wiki/File:"):
            filename = path[len("/wiki/File:"):]
            if not filename:
                return None, "Could not extract Wikimedia filename"

            redirect_url = (
                "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
                + quote(filename, safe="()!,._-'")
            )
            return redirect_url, None

        if "/wiki/Special:Redirect/file/" in path:
            return value.split("#", 1)[0], None

    return None, "Not a supported Wikimedia Commons URL"


def download_and_validate(session, url, destination_base):
    """Download, validate and save an image using its actual file format."""
    response = session.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if not content_type.startswith("image/"):
        raise ValueError(
            f"URL did not return an image (Content-Type: {content_type})"
        )

    image = Image.open(BytesIO(response.content))
    image.verify()

    image_format = (image.format or "").lower()
    extensions = {
        "jpeg": ".jpg",
        "png": ".png",
        "gif": ".gif",
        "webp": ".webp",
        "tiff": ".tiff",
        "bmp": ".bmp",
    }
    extension = extensions.get(image_format, ".img")

    destination = destination_base.with_suffix(extension)
    destination.write_bytes(response.content)

    return destination, len(response.content), response.url


# ------------------------- MAIN -------------------------

def main():
    if not EXCEL_FILE.exists():
        print(f"ERROR: Excel file not found: {EXCEL_FILE.resolve()}")
        print("Place the Excel file beside this script, or edit EXCEL_FILE.")
        sys.exit(1)

    workbook = load_workbook(EXCEL_FILE, data_only=True)
    if SHEET_NAME not in workbook.sheetnames:
        raise ValueError(f"Sheet '{SHEET_NAME}' not found.")

    sheet = workbook[SHEET_NAME]

    # IMPORTANT: setdefault keeps the FIRST occurrence of a header.
    # Sheet1 has a second "Primary_Category" header in a summary table
    # (column H). The actual per-image category is the first one (column E).
    columns = {}
    for cell in sheet[1]:
        if cell.value is not None:
            columns.setdefault(str(cell.value).strip(), cell.column)

    required = ["ID", "Image", "Source", "Primary_Category"]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    id_col = columns["ID"]
    image_col = columns["Image"]
    source_col = columns["Source"]
    category_col = columns["Primary_Category"]

    print("Using these Excel columns:")
    print(f"  ID:              {id_col}")
    print(f"  Image:           {image_col}")
    print(f"  Source:          {source_col}")
    print(f"  Primary_Category:{category_col}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    report_rows = []
    downloaded = skipped = failed = 0

    for row_number in trange(2, sheet.max_row + 1, desc="Processing rows"):
        image_id = safe_id(sheet.cell(row_number, id_col).value, row_number - 1)
        image_value = sheet.cell(row_number, image_col).value
        source = str(sheet.cell(row_number, source_col).value or "").strip()
        category_value = sheet.cell(row_number, category_col).value
        category_name = safe_folder_name(category_value)

        category_dir = OUTPUT_DIR / category_name
        category_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "Excel_Row": row_number,
            "ID": image_id,
            "Primary_Category": category_name,
            "Source": source,
            "Original_Link_or_Value": image_value,
            "Status": "",
            "Saved_File": "",
            "Final_URL": "",
            "Details": "",
        }

        # Personal / Google Photos images are left for manual downloading.
        if source.lower() != "wikimedia":
            report["Status"] = "Skipped - manual download"
            report["Details"] = "Source is not Wikimedia."
            skipped += 1
            report_rows.append(report)
            print(f"[SKIP] ID {image_id}: non-Wikimedia source")
            continue

        download_url, error = get_download_url(image_value)
        if error:
            report["Status"] = "Skipped - invalid link"
            report["Details"] = error
            skipped += 1
            report_rows.append(report)
            print(f"[SKIP] ID {image_id}: {error}")
            continue

        # No duplicate detection here, intentionally.
        # Each spreadsheet row is a separate evaluation entry. Even if the
        # URL is repeated, save another copy under this row's category and ID.
        destination_base = category_dir / f"{image_id}_image"

        # Avoid overwriting a previous successful download for this row.
        existing = list(category_dir.glob(f"{image_id}_image.*"))
        if existing:
            report["Status"] = "Skipped - file already exists"
            report["Saved_File"] = str(existing[0])
            report["Details"] = "File for this Excel ID already exists."
            skipped += 1
            report_rows.append(report)
            print(f"[SKIP] ID {image_id}: file already exists")
            continue

        try:
            saved_path, file_size, final_url = download_and_validate(
                session, download_url, destination_base
            )
            report["Status"] = "Downloaded"
            report["Saved_File"] = str(saved_path)
            report["Final_URL"] = final_url
            report["Details"] = f"{file_size:,} bytes"
            downloaded += 1
            print(f"[OK]   ID {image_id} -> {category_name}/{saved_path.name}")

        except Exception as exc:
            report["Status"] = "Failed"
            report["Final_URL"] = download_url
            report["Details"] = str(exc)
            failed += 1
            print(f"[FAIL] ID {image_id}: {exc}")

        report_rows.append(report)
        time.sleep(REQUEST_DELAY_SECONDS)

    report_path = OUTPUT_DIR / REPORT_NAME
    fieldnames = list(report_rows[0].keys()) if report_rows else []

    with report_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(report_rows)

    print("\n---------------- SUMMARY ----------------")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {failed}")
    print(f"Images folder: {OUTPUT_DIR.resolve()}")
    print(f"Download report: {report_path.resolve()}")
    print("-----------------------------------------")

    session.close()

#%%
if __name__ == "__main__":
    main()