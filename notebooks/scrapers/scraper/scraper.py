#!/usr/bin/env python3
"""
AICC Translation Scraper — async multi-worker edition
======================================================
Reads a CSV containing an 'AICC_translation' column of URLs like:
    https://aicuneiform.com/search?q=P290300

Uses async Playwright with multiple browser workers running in parallel.
Rows are flushed to the output CSV every SAVE_EVERY URLs.

Output CSV columns (trimmed to essentials):
  - p_number                : e.g. P290300
  - section_number          : e.g. "1. Obverse"
  - scraped_transliteration : Akkadian text     (lang-akk)
  - scraped_translation     : Human translation (lang-en) if present,
                              otherwise AI       (lang-ml_en)
  - translation_source      : "human" | "ai" | "none"

Install
-------
    pip install playwright beautifulsoup4 lxml tqdm
    playwright install chromium

Usage
-----
    python scraper.py                   # uses input.csv
    python scraper.py myfile.csv        # custom input
"""

import asyncio
import csv
import json
import logging
import os
import random
import re
import sys
from datetime import datetime

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit all settings here
# ══════════════════════════════════════════════════════════════════════════════

INPUT_CSV        = "input.csv"       # can also be passed as CLI arg
OUTPUT_CSV       = "output.csv"
CHECKPOINT_FILE  = "checkpoint.json"
LOG_FILE         = "scraper.log"
URL_COLUMN       = "AICC_translation"

# ── Parallelism ───────────────────────────────────────────────────────────────
NUM_WORKERS      = 100       # number of parallel browser tabs

# ── Saving ────────────────────────────────────────────────────────────────────
SAVE_EVERY       = 10      # flush rows to CSV (and checkpoint) every N URLs

# ── Timing ────────────────────────────────────────────────────────────────────
DELAY_MIN        = 3.0     # min seconds between navigations per worker
DELAY_MAX        = 6.0     # max seconds between navigations per worker
PAGE_TIMEOUT     = 4000   # ms — max time for page.goto()
WAIT_TIMEOUT     = 4000   # ms — max time to wait for content to appear

# ── Retries ───────────────────────────────────────────────────────────────────
MAX_RETRIES      = 3
RETRY_BACKOFF    = 6       # seconds × attempt number

# ── Resource blocking — resource types to abort (speeds up page loads ~50-70%) ─
BLOCKED_TYPES    = {""}

# ── Output fields (only these columns are written to the CSV) ─────────────────
OUTPUT_FIELDNAMES = [
    "p_number",
    "section_number",
    "scraped_transliteration",
    "scraped_translation",
    "translation_source",
]

# ══════════════════════════════════════════════════════════════════════════════
USER_AGENTS = [
    # Chrome Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/119.0.0.0 Safari/537.36",

    # Chrome Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",

    # Chrome Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",

    # Firefox Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",

    # Firefox Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.2; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13.6; rv:122.0) Gecko/20100101 Firefox/122.0",

    # Safari Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 Version/16.6 Safari/605.1.15",

    # Edge Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",

    # Android Chrome
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/122.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 Chrome/121.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 12; Samsung Galaxy S21) AppleWebKit/537.36 Chrome/120.0.0.0 Mobile Safari/537.36",

    # iPhone Safari
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_7 like Mac OS X) AppleWebKit/605.1.15 Version/16.6 Mobile/15E148 Safari/604.1",

    # iPad
    "Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X) AppleWebKit/605.1.15 Version/17.2 Mobile/15E148 Safari/604.1",

] * 2  # ~48 total


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("aicc_scraper")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint(path: str) -> set:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return set(json.load(f).get("processed_urls", []))
    return set()


def save_checkpoint(path: str, processed: set) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"processed_urls": sorted(processed)}, f, indent=2)


# ── HTML parsing ──────────────────────────────────────────────────────────────

def _line_number(span) -> int:
    for cls in span.get("class", []):
        if cls.startswith("line-"):
            try:
                return int(cls[5:])
            except ValueError:
                pass
    return -1


def _extract_lines(lang_div) -> list:
    lines = []
    for span in lang_div.find_all("span", class_="line"):
        lines.append((_line_number(span), span.get_text(" ", strip=True)))
    lines.sort(key=lambda x: x[0])
    return lines


def _lines_to_str(lines: list) -> str:
    return "\n".join(text for _, text in lines)


def _extract_from_containers(containers, url, p_number, label, logger) -> tuple:
    akk_lines, human_lines, ai_lines = [], [], []
    for c in containers:
        akk_div   = c.find("div", class_="lang-akk")
        human_div = c.find("div", class_="lang-en")
        ai_div    = c.find("div", class_="lang-ml_en")
        if akk_div:   akk_lines.extend(_extract_lines(akk_div))
        if human_div: human_lines.extend(_extract_lines(human_div))
        if ai_div:    ai_lines.extend(_extract_lines(ai_div))

    for lst in (akk_lines, human_lines, ai_lines):
        lst.sort(key=lambda x: x[0])

    translit = _lines_to_str(akk_lines)
    if not akk_lines:
        logger.warning(f"[{url}] {p_number} '{label}': no lang-akk found")

    if human_lines:
        return translit, _lines_to_str(human_lines), "human"
    elif ai_lines:
        return translit, _lines_to_str(ai_lines), "ai"
    else:
        logger.warning(f"[{url}] {p_number} '{label}': no translation found")
        return translit, "", "none"


def _parse_pub_div(pub_div, url: str, logger: logging.Logger) -> list:
    """Parse one <div class='pub'> tablet. Returns list of section row dicts."""
    p_number = ""
    a_tag = pub_div.find("a", href=re.compile(r"#P\d+"))
    if a_tag:
        m = re.search(r"#(P\d+)", a_tag["href"])
        if m:
            p_number = m.group(1)
    if not p_number:
        h1 = pub_div.find("h1", class_="otitle")
        if h1:
            a2 = h1.find("a")
            if a2:
                p_number = a2.get_text(strip=True)

    results  = []
    sections = pub_div.find_all("section", class_="textarea")

    if sections:
        for idx, sec in enumerate(sections, start=1):
            h1     = sec.find("h1")
            header = h1.get_text(strip=True) if h1 else f"Section {idx}"
            label  = f"{idx}. {header}"
            containers = sec.find_all("div", class_="translations-container")
            if not containers:
                logger.warning(f"[{url}] {p_number} '{label}': no translations-container")
                continue
            translit, translation, source = _extract_from_containers(
                containers, url, p_number, label, logger)
            results.append({
                "p_number": p_number, "section_number": label,
                "scraped_transliteration": translit,
                "scraped_translation": translation,
                "translation_source": source,
            })
    else:
        containers = pub_div.find_all("div", class_="translations-container")
        if not containers:
            logger.warning(f"[{url}] {p_number}: no translations-container found")
            return [{"p_number": p_number, "section_number": "1",
                     "scraped_transliteration": "ERROR: no translations-container",
                     "scraped_translation": "ERROR: no translations-container",
                     "translation_source": "none"}]
        translit, translation, source = _extract_from_containers(
            containers, url, p_number, "1", logger)
        results.append({
            "p_number": p_number, "section_number": "1",
            "scraped_transliteration": translit,
            "scraped_translation": translation,
            "translation_source": source,
        })

    return results


def _error_row(msg: str) -> dict:
    return {"p_number": "", "section_number": "1",
            "scraped_transliteration": msg,
            "scraped_translation": msg,
            "translation_source": "none"}


# ── Route blocker ─────────────────────────────────────────────────────────────

async def _block_resources(route):
    """Abort requests for resource types that aren't needed for parsing."""
    if route.request.resource_type in BLOCKED_TYPES:
        await route.abort()
    else:
        await route.continue_()


# ── Async scrape (one URL) ────────────────────────────────────────────────────

async def scrape_url_async(url: str, page, logger: logging.Logger) -> list:
    """Fetch one URL using an async Playwright page. Returns list of row dicts."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # ── 1. Load the page ──────────────────────────────────────────────
            # domcontentloaded is far more reliable than networkidle on SPAs.
            # networkidle waits for the network to go fully quiet — JS-heavy
            # sites never truly reach that state, causing consistent timeouts.
            await page.goto(url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")

            # ── 2. Wait for the root container ────────────────────────────────
            # state="attached" just checks the element exists in the DOM tree;
            # it doesn't require it to be visible or fully styled yet.
            try:
                await page.wait_for_selector(
                    "div.pub",
                    state="attached",
                    timeout=15_000,
                )
            except PlaywrightTimeout:
                msg = "ERROR: timed out waiting for div.pub"
                logger.warning(f"[{url}] {msg} (attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BACKOFF * attempt)
                    continue
                return [_error_row(msg)]

            # ── 3. Wait for translations to render (JS-driven content) ────────
            # Poll with page.wait_for_function so we know the JS has fully
            # populated the DOM before we try to parse it.
            try:
                await page.wait_for_function(
                    """() => document.querySelectorAll(
                            'div.translations-container'
                        ).length > 0""",
                    timeout=20_000,
                )
            except PlaywrightTimeout:
                # Scroll to trigger any lazy-load / intersection-observer
                # behaviour, then wait one more time before giving up.
                logger.warning(f"[{url}] translations-container missing — trying scroll nudge")
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1.5)
                try:
                    await page.wait_for_function(
                        """() => document.querySelectorAll(
                                'div.translations-container'
                            ).length > 0""",
                        timeout=10_000,
                    )
                except PlaywrightTimeout:
                    msg = "ERROR: timed out waiting for translations-container"
                    logger.warning(f"[{url}] {msg}")
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_BACKOFF * attempt)
                        continue
                    return [_error_row(msg)]

            # Page is ready — exit the retry loop
            break

        except PlaywrightTimeout:
            msg = f"ERROR: page load timed out (attempt {attempt}/{MAX_RETRIES})"
            logger.warning(f"[{url}] {msg}")
            if attempt == MAX_RETRIES:
                return [_error_row(msg)]
            await asyncio.sleep(RETRY_BACKOFF * attempt)

        except Exception as exc:
            msg = f"ERROR: {type(exc).__name__} — {exc}"
            logger.warning(f"[{url}] {msg}")
            if attempt == MAX_RETRIES:
                return [_error_row(msg)]
            await asyncio.sleep(RETRY_BACKOFF * attempt)

    # ── 4. Parse ──────────────────────────────────────────────────────────────
    html     = await page.content()
    soup     = BeautifulSoup(html, "html.parser")
    pub_divs = soup.find_all("div", class_="pub")

    if not pub_divs:
        msg = "ERROR: no .pub divs found after JS render"
        logger.warning(f"[{url}] {msg}")
        return [_error_row(msg)]

    all_results = []
    for pub in pub_divs:
        all_results.extend(_parse_pub_div(pub, url, logger))

    logger.info(f"[{url}] OK — {len(pub_divs)} tablet(s), {len(all_results)} row(s)")
    return all_results


# ── Incremental CSV writer ────────────────────────────────────────────────────

class IncrementalWriter:
    """
    Opens the output CSV once and appends rows as they arrive.
    Thread-safe via asyncio lock.
    Only writes OUTPUT_FIELDNAMES columns — all other input columns are dropped.
    """
    def __init__(self, path: str, resume: bool):
        self._lock   = asyncio.Lock()
        mode         = "a" if resume else "w"
        self._f      = open(path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._f, fieldnames=OUTPUT_FIELDNAMES,
                                      extrasaction="ignore")
        if not resume:
            self._writer.writeheader()

    async def write_rows(self, rows: list):
        async with self._lock:
            for row in rows:
                self._writer.writerow(row)
            self._f.flush()

    def close(self):
        self._f.close()


# ── Worker coroutine ──────────────────────────────────────────────────────────

async def worker(worker_id: int, queue: asyncio.Queue, failed_queue: asyncio.Queue,
                 browser, csv_writer: IncrementalWriter,
                 processed_urls: set, checkpoint_lock: asyncio.Lock,
                 pbar, logger: logging.Logger):
    """
    One async worker. Each worker owns its own browser context + page so that:
      - A randomly chosen User-Agent is used per worker.
      - Resource blocking (images, fonts, stylesheets, ...) is applied per page.
      - A crashed/broken page can be recreated without affecting other workers.

    Failed URLs are pushed to failed_queue for a second-pass retry run.
    Delay between requests is adaptive: backs off exponentially on consecutive
    errors, resets to DELAY_MIN once a page succeeds.
    """
    ua      = random.choice(USER_AGENTS)
    context = await browser.new_context(user_agent=ua)
    page    = await context.new_page()
    await page.route("**/*", _block_resources)
    logger.debug(f"Worker {worker_id}: UA → {ua}")

    save_counter       = 0
    consecutive_errors = 0

    try:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            url, original_row = item

            # FIX: scrape_url_async takes (url, page, logger) and returns a list
            sections = await scrape_url_async(url, page, logger)

            # Detect error rows — route to failed_queue for later retry
            is_error = any(r["scraped_transliteration"].startswith("ERROR")
                           for r in sections)
            if is_error:
                consecutive_errors += 1
                await failed_queue.put((url, original_row))
                logger.warning(f"Worker {worker_id}: queued {url} for retry pass")
            else:
                consecutive_errors = 0
                await csv_writer.write_rows(sections)

                async with checkpoint_lock:
                    processed_urls.add(url)
                    save_counter += 1
                    if save_counter % SAVE_EVERY == 0:
                        save_checkpoint(CHECKPOINT_FILE, processed_urls)
                        logger.debug(
                            f"Worker {worker_id}: checkpoint at {save_counter} URLs")

            pbar.update(1)
            queue.task_done()

            # Adaptive delay: exponential back-off on errors, reset on success
            if consecutive_errors == 0:
                delay = random.uniform(DELAY_MIN, DELAY_MAX)
            else:
                delay = min(DELAY_MAX * (2 ** consecutive_errors), 30.0)
            await asyncio.sleep(delay)

    finally:
        await page.close()
        await context.close()


# ── Retry pass (second sweep over failed URLs) ────────────────────────────────

async def retry_pass(failed_queue: asyncio.Queue, browser,
                     csv_writer: IncrementalWriter,
                     processed_urls: set, checkpoint_lock: asyncio.Lock,
                     logger: logging.Logger):
    """Re-attempt all URLs that failed during the main pass, sequentially."""
    if failed_queue.empty():
        return

    retry_items = []
    while not failed_queue.empty():
        retry_items.append(failed_queue.get_nowait())

    logger.info(f"Retry pass: {len(retry_items)} URL(s) to re-attempt")
    print(f"\n  Retry pass: {len(retry_items)} failed URL(s) to re-attempt ...")

    ua      = random.choice(USER_AGENTS)
    context = await browser.new_context(user_agent=ua)
    page    = await context.new_page()
    await page.route("**/*", _block_resources)

    try:
        for url, _ in retry_items:
            await asyncio.sleep(random.uniform(DELAY_MAX, DELAY_MAX * 2))
            # FIX: scrape_url_async takes (url, page, logger) and returns a list
            sections = await scrape_url_async(url, page, logger)
            is_error = any(r["scraped_transliteration"].startswith("ERROR")
                           for r in sections)
            # Write regardless — errors get logged as ERROR rows so nothing is lost
            await csv_writer.write_rows(sections)
            if is_error:
                logger.error(f"Retry failed permanently: {url}")
            else:
                async with checkpoint_lock:
                    processed_urls.add(url)
                logger.info(f"Retry succeeded: {url}")
    finally:
        await page.close()
        await context.close()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(input_csv: str, logger: logging.Logger):
    if not os.path.exists(input_csv):
        print(f"X Input file '{input_csv}' not found.")
        sys.exit(1)

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if URL_COLUMN not in fieldnames:
            print(f"X Column '{URL_COLUMN}' not found. Available: {fieldnames}")
            sys.exit(1)
        rows = list(reader)

    print(f"Loaded {len(rows):,} rows from '{input_csv}'")

    processed_urls = load_checkpoint(CHECKPOINT_FILE)
    resume         = os.path.exists(OUTPUT_CSV) and len(processed_urls) > 0

    if resume:
        print(f"Resuming - {len(processed_urls):,} URLs already done")

    pending = [
        (row.get(URL_COLUMN, "").strip(), row)
        for row in rows
        if row.get(URL_COLUMN, "").strip()
        and row.get(URL_COLUMN, "").strip() not in processed_urls
    ]

    print(
        f"  Total rows   : {len(rows):,}\n"
        f"  Already done : {len(processed_urls):,}\n"
        f"  To scrape    : {len(pending):,}\n"
        f"  Workers      : {NUM_WORKERS}\n"
        f"  Save every   : {SAVE_EVERY} URLs\n"
        f"  Wait timeout : {WAIT_TIMEOUT // 1000}s\n"
        f"  Output cols  : {', '.join(OUTPUT_FIELDNAMES)}\n"
    )

    if not pending:
        print("Nothing left to scrape.")
        return

    queue        = asyncio.Queue()
    failed_queue = asyncio.Queue()
    for item in pending:
        await queue.put(item)

    csv_writer      = IncrementalWriter(OUTPUT_CSV, resume)
    checkpoint_lock = asyncio.Lock()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)

        # ── Main pass ────────────────────────────────────────────────────────
        with tqdm(total=len(pending), desc="Scraping", unit="URL",
                  dynamic_ncols=True) as pbar:
            workers = [
                asyncio.create_task(
                    worker(i, queue, failed_queue, browser, csv_writer,
                           processed_urls, checkpoint_lock, pbar, logger)
                )
                for i in range(NUM_WORKERS)
            ]
            await asyncio.gather(*workers)

        # ── Retry pass ───────────────────────────────────────────────────────
        await retry_pass(failed_queue, browser, csv_writer,
                         processed_urls, checkpoint_lock, logger)

        await browser.close()

    csv_writer.close()

    save_checkpoint(CHECKPOINT_FILE, processed_urls)
    logger.info(f"Final checkpoint saved - {len(processed_urls)} URLs done")

    print(
        f"\nDone!\n"
        f"  Output file : '{OUTPUT_CSV}'\n"
        f"  Log file    : '{LOG_FILE}'\n"
        f"  Checkpoint  : '{CHECKPOINT_FILE}'\n"
    )


def main():
    input_csv = sys.argv[1] if len(sys.argv) > 1 else INPUT_CSV
    logger    = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info(f"Run started: {datetime.now().isoformat()}")
    asyncio.run(main_async(input_csv, logger))


if __name__ == "__main__":
    main()
