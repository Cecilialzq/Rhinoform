#!/usr/bin/env python3
"""Extract and render every page of the submitted PDF for visual audit."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pdfplumber
from PIL import Image, ImageDraw


PDF = Path("/Users/lizequan/Desktop/FYP_revised/最终版report.pdf")
REPO = Path("/Users/lizequan/Downloads/Rhinoform-main")
TMP = REPO / "tmp" / "pdfs" / "report_review"
PDFTOPPM = Path(
    "/Users/lizequan/.cache/codex-runtimes/"
    "codex-primary-runtime/dependencies/bin/override/pdftoppm"
)


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    text_pages: list[str] = []
    with pdfplumber.open(PDF) as doc:
        for page_no, page in enumerate(doc.pages, 1):
            text_pages.append(f"\n===== PAGE {page_no} =====\n{page.extract_text(layout=True) or ''}")
        page_count = len(doc.pages)
    (TMP / "report_pdf_layout.txt").write_text("\n".join(text_pages), encoding="utf-8")

    subprocess.run(
        [str(PDFTOPPM), "-png", "-r", "90", str(PDF), str(TMP / "page")],
        check=True,
    )
    pages = sorted(TMP.glob("page-*.png"))
    if len(pages) != page_count:
        raise RuntimeError(f"rendered {len(pages)} pages, expected {page_count}")

    thumb_w = 420
    margin = 16
    label_h = 28
    per_sheet = 9
    for start in range(0, len(pages), per_sheet):
        group = pages[start : start + per_sheet]
        thumbs: list[Image.Image] = []
        for path in group:
            image = Image.open(path).convert("RGB")
            thumb_h = round(image.height * thumb_w / image.width)
            thumbs.append(image.resize((thumb_w, thumb_h)))
        cell_h = max(im.height for im in thumbs) + label_h
        sheet = Image.new(
            "RGB", (3 * thumb_w + 4 * margin, 3 * cell_h + 4 * margin), "white"
        )
        draw = ImageDraw.Draw(sheet)
        for offset, image in enumerate(thumbs):
            row, col = divmod(offset, 3)
            x = margin + col * (thumb_w + margin)
            y = margin + row * (cell_h + margin)
            page_no = start + offset + 1
            draw.text((x, y), f"PDF page {page_no}", fill="black")
            sheet.paste(image, (x, y + label_h))
        out = TMP / f"contact_{start + 1:03d}_{start + len(group):03d}.jpg"
        sheet.save(out, quality=88)
    print(f"pages={page_count} contact_sheets={(page_count + per_sheet - 1) // per_sheet}")


if __name__ == "__main__":
    main()
