#!/usr/bin/env python3
"""
Download Fed Board speeches as PDFs (2020+) using the same data the website loads:
https://www.federalreserve.gov/json/ne-speeches.json
https://www.federalreserve.gov/json/nespeakers.json

Most speech PDFs live at /newsevents/speech/files/<stem>.pdf where <stem> matches
the .htm basename. If that's missing, the script parses the HTML page for a PDF link.

Politeness: set a descriptive User-Agent; small delay between PDF requests.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime
from html import unescape
from pathlib import Path

BASE = "https://www.federalreserve.gov"
SPEECHES_JSON = f"{BASE}/json/ne-speeches.json"
SPEAKERS_JSON = f"{BASE}/json/nespeakers.json"
# Federalreserve.gov returns 403 for many requests without a normal browser User-Agent.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

_TAG_RE = re.compile(r"<[^>]+>")
_PDF_HREF_RE = re.compile(
    r'href="(/newsevents/speech/files/[^"\'?#]+\.pdf)"', re.I
)


def _fetch_json(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        raw = resp.read().decode("utf-8-sig")
    return json.loads(raw)


def _http_get(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        return resp.status, resp.read()


def _http_head_ok(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError:
        return False
    except OSError:
        return False


def strip_html(s: str) -> str:
    return unescape(_TAG_RE.sub("", s)).strip()


def parse_speech_date(d: str) -> datetime:
    date_part = d.split()[0]
    return datetime.strptime(date_part, "%m/%d/%Y")


def load_speakers() -> list[dict]:
    data = _fetch_json(SPEAKERS_JSON)
    return [x for x in data if isinstance(x, dict) and "name" in x and "label" in x]


def load_speeches() -> list[dict]:
    data = _fetch_json(SPEECHES_JSON)
    if data and isinstance(data[-1], dict) and "updateDate" in data[-1]:
        data = data[:-1]
    return [x for x in data if isinstance(x, dict) and "l" in x]


def classify_folder(
    speech: dict,
    current_pairs: list[tuple[str, str]],
) -> str:
    """
    Map one speech to a folder label (matches website checkbox labels).
    current_pairs: (full name, folder label) for current Board categories only.
    """
    s = strip_html(str(speech.get("s", "")))
    for name, label in sorted(current_pairs, key=lambda p: -len(p[0])):
        if name in s:
            return label
    if speech.get("o") == "yes":
        return "Other Officials"
    return "Former Officials"


def speech_pdf_url_from_stem(stem: str) -> str:
    return f"{BASE}/newsevents/speech/files/{stem}.pdf"


def discover_pdf_url(speech_path: str) -> str | None:
    stem = Path(speech_path).stem
    direct = speech_pdf_url_from_stem(stem)
    if _http_head_ok(direct):
        return direct

    page_url = BASE + speech_path
    try:
        _, body = _http_get(page_url)
    except OSError:
        return None
    text = body.decode("utf-8", errors="replace")
    m = _PDF_HREF_RE.search(text)
    if m:
        return BASE + m.group(1)
    return None


def safe_filename(title: str, speech_date: datetime, stem: str) -> str:
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title)
    base = re.sub(r"\s+", " ", base).strip() or "untitled"
    if len(base) > 120:
        base = base[:120].rstrip()
    d = speech_date.strftime("%Y-%m-%d")
    return f"{d}__{base}__{stem}.pdf"


def download_file(url: str, dest: Path, delay: float) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ctx = ssl.create_default_context()
    time.sleep(delay)
    with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
        dest.write_bytes(resp.read())
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Download Fed speeches as PDFs into FED/ subfolders.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("FED"),
        help="Output root directory (default: ./FED)",
    )
    p.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2023-12-31", help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Seconds to sleep before each PDF GET (default: 0.4)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List actions without downloading",
    )
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23, minute=59, second=59)

    speakers = load_speakers()
    current_pairs = [
        (sp["name"], sp["label"])
        for sp in speakers
        if sp["label"] not in ("Former Officials", "Other Officials")
    ]

    speeches = load_speeches()
    in_range = []
    for sp in speeches:
        try:
            dt = parse_speech_date(sp["d"])
        except (ValueError, KeyError):
            continue
        if start <= dt <= end:
            in_range.append(sp)

    print(f"Found {len(in_range)} speeches between {args.start} and {args.end}")

    missing_pdf: list[str] = []
    for sp in in_range:
        folder = classify_folder(sp, current_pairs)
        dt = parse_speech_date(sp["d"])
        title = str(sp.get("t", "untitled"))
        stem = Path(sp["l"]).stem
        fname = safe_filename(title, dt, stem)
        dest = args.out / folder / fname

        if args.dry_run:
            print(f"[dry-run] {folder}/{fname}")
            continue

        if dest.exists():
            print(f"skip exists {dest}")
            continue

        pdf_url = discover_pdf_url(sp["l"])
        if not pdf_url:
            missing_pdf.append(f"{sp['l']} | {title}")
            print(f"NO PDF  {sp['l']}")
            continue

        try:
            download_file(pdf_url, dest, args.delay)
            print(f"wrote {dest}")
        except Exception as e:
            print(f"FAIL {pdf_url} -> {dest}: {e}")

    if missing_pdf:
        print(f"\nSpeeches without a resolvable PDF ({len(missing_pdf)}):")
        for line in missing_pdf:
            print("  ", line)


if __name__ == "__main__":
    main()
