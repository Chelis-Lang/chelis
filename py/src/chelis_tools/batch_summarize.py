"""Batch download and summarize PDFs from a publications page."""

import argparse
import html.parser
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse, urlunparse

from chelis_tools.openrouter import chat
from chelis_tools.summarize_pdf import SYSTEM_PROMPT, extract_text


class PDFLinkExtractor(html.parser.HTMLParser):
    """Extract all href attributes ending in .pdf from HTML."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value and value.endswith(".pdf"):
                    self.links.append(value)


def scrape_pdf_links(url):
    """Fetch a page and return all PDF links as absolute URLs."""
    req = urllib.request.Request(url, headers={"User-Agent": "chelis-tools/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html_text = resp.read().decode("utf-8")
    parser = PDFLinkExtractor()
    parser.feed(html_text)
    return [urljoin(url, link) for link in parser.links]


def _encode_url(url):
    """Percent-encode non-ASCII characters in URL path."""
    parsed = urlparse(url)
    encoded_path = quote(parsed.path, safe="/")
    return urlunparse(parsed._replace(path=encoded_path))


def download_pdf(url, dest):
    """Download a PDF to a local path."""
    url = _encode_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": "chelis-tools/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())


def pdf_url_to_md_path(url, base_url, output_dir):
    """Convert a PDF URL to a local .md output path preserving directory structure."""
    # Extract the relative path from the base URL
    base = base_url.rsplit("/", 1)[0] + "/"
    if url.startswith(base):
        rel = url[len(base):]
    else:
        rel = url.rsplit("/", 1)[-1]
    return output_dir / re.sub(r"\.pdf$", ".md", rel)


def main():
    parser = argparse.ArgumentParser(description="Batch download and summarize PDFs")
    parser.add_argument("url", help="URL of publications page to scrape")
    parser.add_argument("-o", "--output", required=True, help="Output directory for markdown files")
    parser.add_argument("--subdir", help="Only process PDFs under this subdir (e.g. 'publications' or 'student-projects')")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API calls in seconds (default: 2)")
    parser.add_argument("--force", action="store_true", help="Re-summarize even if .md already exists")
    parser.add_argument("-m", "--model", default=None, help="Preferred model (default: use fallback chain)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scraping PDF links from {args.url}...", file=sys.stderr)
    pdf_urls = scrape_pdf_links(args.url)
    print(f"Found {len(pdf_urls)} PDFs", file=sys.stderr)

    if args.subdir:
        pdf_urls = [u for u in pdf_urls if f"/{args.subdir}/" in u]
        print(f"Filtered to {len(pdf_urls)} PDFs matching subdir '{args.subdir}'", file=sys.stderr)

    succeeded = 0
    skipped = 0
    failed = 0

    for i, url in enumerate(pdf_urls, 1):
        md_path = pdf_url_to_md_path(url, args.url, output_dir)

        if md_path.exists() and not args.force:
            print(f"[{i}/{len(pdf_urls)}] SKIP {md_path.name} (exists)", file=sys.stderr)
            skipped += 1
            continue

        print(f"[{i}/{len(pdf_urls)}] {url}", file=sys.stderr)

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                tmp_path = Path(tmp.name)
                print(f"  Downloading...", file=sys.stderr)
                download_pdf(url, tmp_path)

                print(f"  Extracting text...", file=sys.stderr)
                text = extract_text(str(tmp_path))
                if not text.strip():
                    print(f"  WARNING: no text extracted, skipping", file=sys.stderr)
                    failed += 1
                    continue

                print(f"  Summarizing ({len(text)} chars)...", file=sys.stderr)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ]
                kwargs = {}
                if args.model:
                    from chelis_tools.openrouter import MODELS
                    kwargs["model"] = MODELS.get(args.model, args.model)
                result = chat(messages, **kwargs)

                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(result + "\n")
                print(f"  -> {md_path}", file=sys.stderr)
                succeeded += 1

        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failed += 1

        if i < len(pdf_urls) and args.delay > 0:
            time.sleep(args.delay)

    print(f"\nDone: {succeeded} summarized, {skipped} skipped, {failed} failed", file=sys.stderr)


if __name__ == "__main__":
    main()
