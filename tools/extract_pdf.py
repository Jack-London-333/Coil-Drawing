"""Extract scanned patent PDF pages as PNG images for formula verification."""
import sys
from pathlib import Path

import fitz  # pymupdf

SRC = Path(r"C:\Users\19144\Desktop\CN104965948B.pdf")
OUT = Path(__file__).resolve().parent.parent / "docs" / "patent_pages"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(SRC)
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=200)
        out = OUT / f"page_{i:02d}.png"
        pix.save(out)
        print(f"saved {out} ({pix.width}x{pix.height})")
    print(f"total pages: {doc.page_count}")


if __name__ == "__main__":
    sys.exit(main())
