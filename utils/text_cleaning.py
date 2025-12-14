import re

def normalize_whitespace(text: str) -> str:
    """Replaces multiple whitespace characters with a single space and strips leading/trailing space."""
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text: str) -> str:
    """Performs general text cleaning, including replacing non-breaking spaces."""
    # Replace non-breaking space with regular space
    text = text.replace("\u00a0", " ")
    return normalize_whitespace(text)