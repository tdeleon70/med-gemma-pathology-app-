
import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from a PDF file provided as bytes."""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        return f"Error processing PDF: {e}"

def parse_csv(file):
    """Parses a CSV file and returns a pandas DataFrame."""
    try:
        return pd.read_csv(file)
    except Exception as e:
        return f"Error processing CSV: {e}"
