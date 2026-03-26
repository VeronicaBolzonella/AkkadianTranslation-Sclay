#!/usr/bin/env python3
"""
AICC Browser Scraper — sequential navigation edition
=====================================================
Navigates https://aicuneiform.com/akk (or any AICC browse URL) by clicking
the ">" next button, scraping each tablet as it appears, and writing rows to
a CSV.  Resumes automatically from a checkpoint.

Output CSV columns:
  - p_number                : e.g. P290300
  - section_number          : e.g. "1. Obverse"
  - scraped_transliteration : Akkadian text     (lang-akk)
  - scraped_translation     : Human translation (lang-en) if present,
                              otherwise AI       (lang-ml_en)
  - translation_source      : "human" | "ai" | "none"

Install
-------
    pip install playwright beautifulsoup4 tqdm
    playwright install chromium

Usage
-----
    python scraper_browse.py               # starts from the beginning
    python scraper_browse.py --resume      # resumes from checkpoint
    python scraper_browse.py --start 500   # jump to position 500 first
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
from datetime import datetime

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit all settings here
# ══════════════════════════════════════════════════════════════════════════════

START_URL = "https://aicuneiform.com/akk"  # browse page to start from
OUTPUT_CSV = "output_browse.csv"
CHECKPOINT_FILE = "checkpoint_browse.json"
LOG_FILE = "scraper_browse.log"

# ── Timing ────────────────────────────────────────────────────────────────────
PAGE_LOAD_TIMEOUT = 30_000  # ms — initial page load
CONTENT_TIMEOUT = 20_000  # ms — wait for tablet content after click
NEXT_CLICK_DELAY = 1.5  # seconds — pause after each click (be polite)

# ── Saving ────────────────────────────────────────────────────────────────────
SAVE_EVERY = 10  # flush CSV and checkpoint every N tablets

# ── Retries ───────────────────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds × attempt number

# ── Display ───────────────────────────────────────────────────────────────────
HEADLESS = True  # set False to watch the browser

# ── Resource blocking ─────────────────────────────────────────────────────────
BLOCKED_TYPES = {"image", "media", "font", "stylesheet", "other"}

# ── Output fields ─────────────────────────────────────────────────────────────
OUTPUT_FIELDNAMES = [
    "position",
    "p_number",
    "section_number",
    "scraped_transliteration",
    "scraped_translation",
    "translation_source",
]

# ══════════════════════════════════════════════════════════════════════════════


# ── Logging ───────────────────────────────────────────────────────────────────


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("aicc_browse_scraper")
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


def load_checkpoint(path: str) -> dict:
    """Returns {"position": int, "scraped_p_numbers": [...]}"""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            return {
                "position": data.get("position", 0),
                "scraped_p_numbers": set(data.get("scraped_p_numbers", [])),
            }
    return {"position": 0, "scraped_p_numbers": set()}


def save_checkpoint(path: str, position: int, scraped_p_numbers: set) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "position": position,
                "scraped_p_numbers": sorted(scraped_p_numbers),
            },
            f,
            indent=2,
        )


# ── HTML parsing (reused from original scraper) ───────────────────────────────


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


def _extract_from_containers(containers, p_number, label, logger) -> tuple:
    akk_lines, human_lines, ai_lines = [], [], []
    for c in containers:
        akk_div = c.find("div", class_="lang-akk")
        human_div = c.find("div", class_="lang-en")
        ai_div = c.find("div", class_="lang-ml_en")
        if akk_div:
            akk_lines.extend(_extract_lines(akk_div))
        if human_div:
            human_lines.extend(_extract_lines(human_div))
        if ai_div:
            ai_lines.extend(_extract_lines(ai_div))

    for lst in (akk_lines, human_lines, ai_lines):
        lst.sort(key=lambda x: x[0])

    translit = _lines_to_str(akk_lines)
    if not akk_lines:
        logger.warning(f"{p_number} '{label}': no lang-akk found")

    if human_lines:
        return translit, _lines_to_str(human_lines), "human"
    elif ai_lines:
        return translit, _lines_to_str(ai_lines), "ai"
    else:
        logger.warning(f"{p_number} '{label}': no translation found")
        return translit, "", "none"


def _parse_pub_div(pub_div, logger: logging.Logger) -> list:
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

    results = []
    sections = pub_div.find_all("section", class_="textarea")

    if sections:
        for idx, sec in enumerate(sections, start=1):
            h1 = sec.find("h1")
            header = h1.get_text(strip=True) if h1 else f"Section {idx}"
            label = f"{idx}. {header}"
            containers = sec.find_all("div", class_="translations-container")
            if not containers:
                logger.warning(f"{p_number} '{label}': no translations-container")
                continue
            translit, translation, source = _extract_from_containers(
                containers, p_number, label, logger
            )
            results.append(
                {
                    "p_number": p_number,
                    "section_number": label,
                    "scraped_transliteration": translit,
                    "scraped_translation": translation,
                    "translation_source": source,
                }
            )
    else:
        containers = pub_div.find_all("div", class_="translations-container")
        if not containers:
            logger.warning(f"{p_number}: no translations-container found")
            return [
                {
                    "p_number": p_number,
                    "section_number": "1",
                    "scraped_transliteration": "ERROR: no translations-container",
                    "scraped_translation": "ERROR: no translations-container",
                    "translation_source": "none",
                }
            ]
        translit, translation, source = _extract_from_containers(
            containers, p_number, "1", logger
        )
        results.append(
            {
                "p_number": p_number,
                "section_number": "1",
                "scraped_transliteration": translit,
                "scraped_translation": translation,
                "translation_source": source,
            }
        )

    return results


def _error_rows(p_number: str, msg: str) -> list:
    return [
        {
            "p_number": p_number,
            "section_number": "1",
            "scraped_transliteration": msg,
            "scraped_translation": msg,
            "translation_source": "none",
        }
    ]


# ── Resource blocker ──────────────────────────────────────────────────────────


async def _block_resources(route):
    if route.request.resource_type in BLOCKED_TYPES:
        await route.abort()
    else:
        await route.continue_()


# ── Page helpers ──────────────────────────────────────────────────────────────


async def _wait_for_tablet(page, logger: logging.Logger) -> bool:
    """
    Wait until a .pub div with a translations-container is present.
    Returns True on success, False on timeout.
    """
    try:
        await page.wait_for_selector(
            "div.pub", state="attached", timeout=CONTENT_TIMEOUT
        )
    except PlaywrightTimeout:
        logger.warning("Timed out waiting for div.pub")
        return False

    try:
        await page.wait_for_function(
            "() => document.querySelectorAll('div.translations-container').length > 0",
            timeout=CONTENT_TIMEOUT,
        )
        return True
    except PlaywrightTimeout:
        # Try a scroll nudge
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1.0)
        try:
            await page.wait_for_function(
                "() => document.querySelectorAll('div.translations-container').length > 0",
                timeout=10_000,
            )
            return True
        except PlaywrightTimeout:
            logger.warning("Timed out waiting for translations-container")
            return False


def _read_position(page_sync_html: str) -> tuple[int, int]:
    """
    Extract current position and total from the selection-text span.
    e.g. "149 / 30410"  →  (149, 30410)
    Returns (-1, -1) if not found.
    """
    soup = BeautifulSoup(page_sync_html, "html.parser")
    span = soup.find("span", class_="selection-text")
    if span:
        m = re.search(r"(\d+)\s*/\s*(\d+)", span.get_text())
        if m:
            return int(m.group(1)), int(m.group(2))
    return -1, -1


async def _get_position(page) -> tuple[int, int]:
    html = await page.content()
    return _read_position(html)


async def _scrape_current(page, position: int, logger: logging.Logger) -> list:
    """Parse the tablet currently shown on the page."""
    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")
    pub_divs = soup.find_all("div", class_="pub")

    if not pub_divs:
        msg = "ERROR: no .pub divs found"
        logger.warning(f"Position {position}: {msg}")
        return _error_rows("", msg)

    all_rows = []
    for pub in pub_divs:
        rows = _parse_pub_div(pub, logger)
        for row in rows:
            row["position"] = position
        all_rows.extend(rows)

    return all_rows


async def _click_next(page, logger: logging.Logger) -> bool:
    """
    Click the > button and wait for the tablet to update.
    Returns True if we successfully advanced to a new tablet.
    """
    # Snapshot current p_number to detect change
    html_before = await page.content()
    soup_before = BeautifulSoup(html_before, "html.parser")
    pub_before = soup_before.find("div", class_="pub")
    title_before = pub_before.get_text()[:60] if pub_before else ""

    # Find and click the ">" button (second .select-button)
    buttons = await page.query_selector_all("button.select-button")
    next_btn = None
    for btn in buttons:
        text = (await btn.inner_text()).strip()
        if text == ">":
            next_btn = btn
            break

    if next_btn is None:
        logger.warning("Could not find the '>' next button")
        return False

    await next_btn.click()
    await asyncio.sleep(0.3)  # brief pause to let JS fire

    # Wait for the content to update (new p_number or new content)
    for _ in range(20):  # up to ~4s
        await asyncio.sleep(0.2)
        html_after = await page.content()
        soup_after = BeautifulSoup(html_after, "html.parser")
        pub_after = soup_after.find("div", class_="pub")
        title_after = pub_after.get_text()[:60] if pub_after else ""
        if title_after != title_before:
            return True

    # Content didn't change — might be last tablet or slow render
    logger.warning("Content did not change after clicking next")
    return False


# ── CSV writer ────────────────────────────────────────────────────────────────


class CsvWriter:
    def __init__(self, path: str, resume: bool):
        mode = "a" if resume else "w"
        self._f = open(path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._f, fieldnames=OUTPUT_FIELDNAMES, extrasaction="ignore"
        )
        if not resume:
            self._writer.writeheader()

    def write_rows(self, rows: list):
        for row in rows:
            self._writer.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()


# ── Main loop ─────────────────────────────────────────────────────────────────


async def main_async(resume: bool, start_position: int, logger: logging.Logger):
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    resume_pos = checkpoint["position"]
    scraped = checkpoint["scraped_p_numbers"]

    # Decide starting position
    if resume and resume_pos > 0:
        start_position = resume_pos
        print(
            f"✓ Resuming from position {start_position} ({len(scraped):,} tablets already done)"
        )
    elif start_position > 0:
        print(f"  Starting at position {start_position}")
    else:
        print("  Starting from the beginning")

    writer = CsvWriter(OUTPUT_CSV, resume=(resume and resume_pos > 0))
    counter = 0  # tablets scraped this session

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=HEADLESS, slow_mo=0)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()
        await page.route("**/*", _block_resources)

        # ── Load the browse page ──────────────────────────────────────────────
        print(f"  Loading {START_URL} ...")
        await page.goto(
            START_URL, timeout=PAGE_LOAD_TIMEOUT, wait_until="domcontentloaded"
        )
        ok = await _wait_for_tablet(page, logger)
        if not ok:
            print("✗ Failed to load the initial page.")
            await browser.close()
            writer.close()
            return

        pos, total = await _get_position(page)
        if total <= 0:
            print("✗ Could not read tablet count from page.")
            await browser.close()
            writer.close()
            return

        print(f"  Page loaded. Current position: {pos} / {total}")

        # ── Fast-forward to resume position using the input field if needed ───
        # The page has a counter "149 / 30410" and two buttons.
        # To jump to an arbitrary position efficiently, we type into the
        # selection-text span — but it's not an input.  Instead we spam the
        # next button in a quick loop without scraping until we reach the
        # target.  For large jumps this could be slow; an alternative is to
        # reload with a URL hash if the site supports it.
        if start_position > pos:
            skip = start_position - pos
            print(f"  Fast-forwarding {skip} positions (this may take a moment)...")
            with tqdm(total=skip, desc="Skipping", unit="tablet") as pbar:
                for _ in range(skip):
                    buttons = await page.query_selector_all("button.select-button")
                    for btn in buttons:
                        if (await btn.inner_text()).strip() == ">":
                            await btn.click()
                            break
                    await asyncio.sleep(0.05)  # very fast — no scraping
                    pbar.update(1)
            await asyncio.sleep(1.0)
            await _wait_for_tablet(page, logger)
            pos, _ = await _get_position(page)
            print(f"  Now at position {pos}")

        # ── Main scrape loop ──────────────────────────────────────────────────
        with tqdm(
            total=total - pos, desc="Scraping", unit="tablet", dynamic_ncols=True
        ) as pbar:
            while True:
                pos, total = await _get_position(page)

                # -- Scrape current tablet ------------------------------------
                rows = await _scrape_current(page, pos, logger)
                p_number = rows[0]["p_number"] if rows else ""

                # Write rows (skip if already done in a previous run)
                if p_number and p_number in scraped:
                    logger.debug(
                        f"Position {pos}: {p_number} already scraped — skipping"
                    )
                else:
                    writer.write_rows(rows)
                    if p_number:
                        scraped.add(p_number)

                counter += 1
                pbar.update(1)
                pbar.set_postfix({"p": p_number, "pos": pos})

                # -- Checkpoint -----------------------------------------------
                if counter % SAVE_EVERY == 0:
                    save_checkpoint(CHECKPOINT_FILE, pos, scraped)
                    logger.debug(f"Checkpoint saved at position {pos}")

                # -- Check if we've reached the end ----------------------------
                if pos >= total:
                    print(f"\n  Reached the last tablet ({pos}/{total}).")
                    break

                # -- Click next -----------------------------------------------
                await asyncio.sleep(NEXT_CLICK_DELAY)

                for attempt in range(1, MAX_RETRIES + 1):
                    advanced = await _click_next(page, logger)
                    if advanced:
                        break
                    logger.warning(f"Click-next attempt {attempt}/{MAX_RETRIES} failed")
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_BACKOFF * attempt)
                else:
                    # Could not advance — log and stop
                    logger.error(f"Gave up advancing from position {pos}")
                    print(f"\n  Could not advance from position {pos}. Stopping.")
                    break

        await browser.close()

    writer.close()

    # Final checkpoint
    pos, _ = _read_position("")  # pos already set in loop
    save_checkpoint(CHECKPOINT_FILE, pos, scraped)

    print(
        f"\n✓ Done!\n"
        f"  Tablets scraped  : {counter:,}\n"
        f"  Output file      : '{OUTPUT_CSV}'\n"
        f"  Log file         : '{LOG_FILE}'\n"
        f"  Checkpoint       : '{CHECKPOINT_FILE}'\n"
    )


def main():
    parser = argparse.ArgumentParser(description="AICC browse-page scraper")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Jump to this position before scraping (1-based)",
    )
    args = parser.parse_args()

    logger = setup_logging(LOG_FILE)
    logger.info("=" * 60)
    logger.info(f"Run started: {datetime.now().isoformat()}")

    asyncio.run(
        main_async(
            resume=args.resume,
            start_position=args.start,
            logger=logger,
        )
    )


if __name__ == "__main__":
    main()
