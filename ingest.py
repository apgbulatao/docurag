"""
ingest.py — Generic docs-site ingestion for a self-hosted RAG pipeline.

Crawls a documentation site with Crawl4AI, converts pages to clean markdown,
chunks by header, and writes JSONL ready for embedding.

Usage:
    python ingest.py --source-url https://docs.astro.build --collection astro
    python ingest.py --source-url https://doc.rust-lang.org/book --collection rust

Requires:
    pip install crawl4ai
    playwright install chromium   # crawl4ai uses Playwright under the hood
"""

import argparse
import asyncio
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import httpx
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig


# ---------------------------------------------------------------------------
# 1. Discover URLs to crawl (prefer sitemap.xml, fall back to link crawling)
# ---------------------------------------------------------------------------

async def get_urls_from_sitemap(base_url: str) -> list[str]:
    """Try the standard sitemap.xml location first — fastest, most reliable."""
    sitemap_url = urljoin(base_url, "/sitemap.xml")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(sitemap_url)
            resp.raise_for_status()
    except httpx.HTTPError:
        return []

    root = ElementTree.fromstring(resp.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text for loc in root.findall(".//sm:loc", ns) if loc.text]

    # Keep only URLs under the same path as the source (avoids blog/marketing pages)
    base_path = urlparse(base_url).path.rstrip("/")
    same_section = [u for u in urls if urlparse(u).path.startswith(base_path)]
    return same_section or urls


async def discover_urls(source_url: str, max_pages: int) -> list[str]:
    urls = await get_urls_from_sitemap(source_url)
    if urls:
        print(f"Found {len(urls)} URLs via sitemap.xml")
        return urls[:max_pages]

    # Fallback: crawl4ai can do a shallow BFS crawl from the root page,
    # following in-domain links. Good enough when there's no sitemap.
    print("No sitemap found, falling back to link crawling from root page")
    domain = urlparse(source_url).netloc
    seen, queue, collected = set(), [source_url], []

    async with AsyncWebCrawler() as crawler:
        while queue and len(collected) < max_pages:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            result = await crawler.arun(url=url)
            if not result.success:
                continue
            collected.append(url)
            for link in result.links.get("internal", []):
                href = link.get("href", "")
                if urlparse(href).netloc in ("", domain) and href not in seen:
                    queue.append(urljoin(url, href))

    return collected


# ---------------------------------------------------------------------------
# 2. Crawl each page and pull clean markdown
# ---------------------------------------------------------------------------

async def crawl_pages(urls: list[str]) -> list[dict]:
    pages = []
    config = CrawlerRunConfig(
        excluded_tags=["nav", "footer", "header", "aside"],
        exclude_external_links=True,
        word_count_threshold=10,  # skip near-empty pages
    )

    async with AsyncWebCrawler() as crawler:
        for i, url in enumerate(urls, 1):
            result = await crawler.arun(url=url, config=config)
            if result.success and result.markdown:
                pages.append({"url": url, "markdown": result.markdown})
                print(f"[{i}/{len(urls)}] ok  {url}")
            else:
                print(f"[{i}/{len(urls)}] skip {url}")
    return pages


# ---------------------------------------------------------------------------
# 3. Chunk by markdown headers (keeps code blocks and sections intact)
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


def chunk_by_headers(markdown: str, url: str, min_chunk_len: int = 100) -> list[dict]:
    matches = list(HEADER_RE.finditer(markdown))
    if not matches:
        text = markdown.strip()
        return [{"heading": None, "text": text, "url": url}] if len(text) >= min_chunk_len else []

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        heading = match.group(2).strip()
        body = markdown[start:end].strip()
        if len(body) >= min_chunk_len:
            chunks.append({"heading": heading, "text": body, "url": url})
    return chunks


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

async def main(source_url: str, collection: str, max_pages: int, out_dir: str):
    urls = await discover_urls(source_url, max_pages)
    if not urls:
        print("No URLs discovered — check the source URL.")
        return

    pages = await crawl_pages(urls)

    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_by_headers(page["markdown"], page["url"]))

    out_path = Path(out_dir) / f"{collection}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            chunk["collection"] = collection
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(pages)} pages -> {len(all_chunks)} chunks -> {out_path}")
    print("Next step: embed each chunk's 'text' field and load into your vector store.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a docs site into chunked markdown JSONL.")
    parser.add_argument("--source-url", required=True, help="Root URL of the docs site, e.g. https://docs.astro.build")
    parser.add_argument("--collection", required=True, help="Name for this collection, e.g. astro, rust")
    parser.add_argument("--max-pages", type=int, default=300, help="Safety cap on number of pages to crawl")
    parser.add_argument("--out-dir", default="./data", help="Output directory for the JSONL file")
    args = parser.parse_args()

    asyncio.run(main(args.source_url, args.collection, args.max_pages, args.out_dir))
