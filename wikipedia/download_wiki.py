"""Download the plain text of every Wikipedia article under one category.

Walks the category tree breadth-first (so the articles closest to the root
category come first), then fetches each article as plain text and appends it
to a JSON-lines file. Re-running resumes where the last run stopped.

    python download_wiki.py --category "History of Canada" --max-pages 3000
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import time
from collections import deque
from pathlib import Path
from typing import Any

import httpx

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "tiny-llms/0.1 (https://github.com/meetrickyb/tiny-llms; educational project)"

# Everything from the first of these headings onward is link lists and
# citations rather than prose, so it is cut off.
TAIL_SECTIONS = re.compile(
    r"^==\s*(See also|References|Notes|Footnotes|Citations|Bibliography|"
    r"Further reading|External links|Sources|Works cited)\s*==\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def api_get(client: httpx.Client, params: dict[str, Any], retries: int = 5) -> dict[str, Any]:
    """Call the MediaWiki API, backing off when the servers ask us to."""
    params = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
    for attempt in range(retries):
        try:
            response = client.get(API_URL, params=params)
        except httpx.TransportError:
            time.sleep(2**attempt)
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(float(response.headers.get("Retry-After", 2**attempt)))
            continue
        response.raise_for_status()
        payload = response.json()
        if payload.get("error", {}).get("code") == "maxlag":
            time.sleep(float(response.headers.get("Retry-After", 5)))
            continue
        return payload
    raise RuntimeError(f"Wikipedia API kept failing for {params}")


def crawl_category(
    client: httpx.Client, root: str, max_depth: int, max_pages: int
) -> list[tuple[int, str]]:
    """Return (pageid, title) for articles under `root`, nearest categories first."""
    pages: dict[int, str] = {}
    seen = {f"Category:{root}"}
    queue: deque[tuple[str, int]] = deque([(f"Category:{root}", 0)])

    while queue and len(pages) < max_pages:
        category, depth = queue.popleft()
        cont: dict[str, str] = {}
        while True:
            payload = api_get(
                client,
                {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": category,
                    "cmtype": "page|subcat",
                    "cmlimit": "max",
                    **cont,
                },
            )
            for member in payload["query"]["categorymembers"]:
                if member["ns"] == 0:
                    pages.setdefault(member["pageid"], member["title"])
                elif member["ns"] == 14 and depth < max_depth and member["title"] not in seen:
                    seen.add(member["title"])
                    queue.append((member["title"], depth + 1))
            if "continue" not in payload:
                break
            cont = payload["continue"]
        print(f"\r  {len(seen)} categories seen, {len(pages)} articles found", end="", flush=True)

    print()
    return list(pages.items())[:max_pages]


def fetch_text(client: httpx.Client, pageid: int) -> str:
    """Fetch one article as plain text, with the reference sections removed."""
    payload = api_get(
        client,
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "wiki",
            "pageids": pageid,
        },
    )
    text = payload["query"]["pages"][0].get("extract", "")
    tail = TAIL_SECTIONS.search(text)
    if tail:
        text = text[: tail.start()]
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--category", default="History of Canada")
    parser.add_argument("--max-depth", type=int, default=3, help="subcategory levels to follow")
    parser.add_argument("--max-pages", type=int, default=3000)
    parser.add_argument("--min-chars", type=int, default=1000, help="skip stubs shorter than this")
    parser.add_argument("--out", type=Path, default=Path("data/articles.jsonl"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if args.out.exists():
        with args.out.open(encoding="utf-8") as f:
            done = {json.loads(line)["pageid"] for line in f}

    # The system trust store (not certifi's bundle) so this also works on
    # networks that inspect TLS with their own root certificate.
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, verify=ssl.create_default_context(), timeout=30
    ) as client:
        print(f"Crawling Category:{args.category} (depth {args.max_depth})")
        pages = crawl_category(client, args.category, args.max_depth, args.max_pages)
        todo = [(pageid, title) for pageid, title in pages if pageid not in done]
        print(f"{len(pages)} articles listed, {len(todo)} still to download")

        kept = chars = 0
        with args.out.open("a", encoding="utf-8") as f:
            for i, (pageid, title) in enumerate(todo, 1):
                text = fetch_text(client, pageid)
                # Stubs are recorded with empty text so a resumed run skips them.
                if len(text) < args.min_chars:
                    text = ""
                else:
                    kept += 1
                    chars += len(text)
                f.write(json.dumps({"pageid": pageid, "title": title, "text": text}) + "\n")
                f.flush()
                print(f"\r  {i}/{len(todo)} fetched, {kept} kept, {chars / 1e6:.1f} MB", end="", flush=True)
                time.sleep(0.05)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
