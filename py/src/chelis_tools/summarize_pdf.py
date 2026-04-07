"""Extract text from a PDF and summarize it via OpenRouter."""

import argparse
import sys
from pathlib import Path

import fitz  # pymupdf

from chelis_tools.openrouter import DEFAULT_MODEL, MODELS, chat


def extract_text(pdf_path):
    """Extract all text from a PDF using pymupdf."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


SYSTEM_PROMPT = """\
You are a research assistant. Summarize the following academic paper in structured markdown.

Use this format:
# <Paper Title>

## Metadata
- **Authors:**
- **Venue/Year:**

## Summary
A 2-3 paragraph high-level summary of what the paper does and why it matters.

## Key Contributions
Bulleted list of the main contributions.

## Technical Approach
Describe the core technical ideas, methods, or algorithms.

## Results
Summarize the evaluation, benchmarks, or experimental findings.

## Relevance
Why this paper might be relevant to language design, compilers, or ML systems work.
"""


def main():
    parser = argparse.ArgumentParser(description="Summarize a PDF via OpenRouter")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("-o", "--output", help="Output markdown file (default: stdout)")
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"Preferred model (default: {DEFAULT_MODEL}). Shortcuts: {', '.join(MODELS.keys())}. Falls through to other models on failure.",
    )
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="Disable automatic model fallback",
    )
    args = parser.parse_args()

    model = MODELS.get(args.model, args.model)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting text from {pdf_path}...", file=sys.stderr)
    text = extract_text(pdf_path)
    if not text.strip():
        print(f"Error: no text extracted from {pdf_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Extracted {len(text)} characters from {pdf_path}", file=sys.stderr)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    result = chat(messages, model=model, fallback=not args.no_fallback)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result + "\n")
        print(f"Written to {out}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
