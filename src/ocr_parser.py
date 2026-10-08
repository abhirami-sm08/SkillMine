from pathlib import Path
from typing import BinaryIO, Union


def extract_text_from_upload(file_obj: Union[str, Path, BinaryIO], suffix: str = "") -> str:
    """Extract text from PDFs and common image formats using optional local OCR dependencies."""
    path = Path(file_obj) if isinstance(file_obj, (str, Path)) else None
    extension = (suffix or (path.suffix if path else "")).lower()

    if extension == ".pdf":
        return _extract_pdf_text(file_obj)
    if extension in {".png", ".jpg", ".jpeg"}:
        return _extract_image_text(file_obj)
    raise ValueError(f"Unsupported upload type: {extension or 'unknown'}")


def _extract_pdf_text(file_obj: Union[str, Path, BinaryIO]) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("Install pdfplumber to parse PDF offer letters.") from exc

    parts = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def _extract_image_text(file_obj: Union[str, Path, BinaryIO]) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install pytesseract and pillow to OCR image offer letters.") from exc

    image = Image.open(file_obj)
    return pytesseract.image_to_string(image).strip()
