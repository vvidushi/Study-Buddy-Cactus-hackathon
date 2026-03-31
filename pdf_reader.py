"""
pdf_reader.py — On-device PDF text extraction and search
=========================================================
Extracts text from uploaded study PDFs and finds the most
relevant passages for a given exam question.
No cloud, no external APIs — pure local processing.
"""

import re
from pathlib import Path


def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    Returns plain text string.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except ImportError:
        raise RuntimeError("pypdf not installed. Run: pip install pypdf")
    except Exception as e:
        raise RuntimeError(f"Could not read PDF: {e}")


def get_relevant_context(full_text: str, question: str, max_chars: int = 1500) -> str:
    """
    Find the most relevant section of the PDF for a given question.

    Strategy:
    1. Skip front matter (first 10% of text — usually metadata/EPUB headers)
    2. Split into paragraphs
    3. Score by: keyword density + keyword count + paragraph quality
    4. Return top-scoring paragraphs up to max_chars, keeping original order
    """
    if not full_text.strip():
        return ""

    # Skip front matter (copyright pages, EPUB headers, etc.)
    skip = max(500, len(full_text) // 20)
    body = full_text[skip:]

    if not question.strip():
        return body[:max_chars].strip()

    # Extract keywords from question (ignore stop words)
    stop_words = {
        "a","an","the","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","what","when","where","who",
        "which","how","why","and","or","but","in","on","at","to",
        "for","of","with","by","from","about","this","that","these",
        "those","my","your","his","her","its","our","their","i","you",
        "he","she","it","we","they","me","him","us","them","not","no",
        "s","t","don","isn","can","just","also","very","more","than"
    }
    keywords = set(
        w.lower() for w in re.findall(r'\b\w+\b', question)
        if w.lower() not in stop_words and len(w) > 2
    )

    if not keywords:
        return body[:max_chars].strip()

    # Split into paragraphs (keep at least 60 chars)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', body) if len(p.strip()) > 60]
    if not paragraphs:
        return body[:max_chars].strip()

    # Score: keyword hits weighted by density (hits / words)
    def score(para: str) -> float:
        words = re.findall(r'\b\w+\b', para.lower())
        if not words:
            return 0.0
        hits = sum(1 for w in words if w in keywords)
        density = hits / len(words)
        # Boost paragraphs that have multiple different keywords
        unique_hits = sum(1 for kw in keywords if kw in para.lower())
        return hits + density * 10 + unique_hits * 2

    # Keep top 5 scoring paragraphs in their original order
    indexed = sorted(enumerate(paragraphs), key=lambda x: score(x[1]), reverse=True)
    top_indices = sorted(i for i, _ in indexed[:5] if score(paragraphs[i]) > 0)

    collected = []
    total = 0
    for i in top_indices:
        para = paragraphs[i]
        if total + len(para) > max_chars:
            remaining = max_chars - total
            if remaining > 150:
                collected.append(para[:remaining] + "…")
            break
        collected.append(para)
        total += len(para)

    return "\n\n".join(collected) if collected else body[:max_chars].strip()


# ── Smoke test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_reader.py <file.pdf> [question]")
    else:
        text = extract_text(sys.argv[1])
        question = sys.argv[2] if len(sys.argv) > 2 else ""
        ctx = get_relevant_context(text, question)
        print(f"Extracted {len(text)} chars total")
        print(f"Relevant context ({len(ctx)} chars):\n")
        print(ctx[:500], "...")
