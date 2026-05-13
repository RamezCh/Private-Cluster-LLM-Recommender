"""OpenCompass leaderboard scraper for general and academic benchmarks."""

import time
import json
import re
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from loguru import logger

from src.config import (
    OPENCOMPASS_CONFIG,
    SELENIUM_CONFIG,
    PROPRIETARY_FILTER,
    TEMP_OPENCOMPASS_GENERAL,
    TEMP_OPENCOMPASS_ACADEMIC,
    is_proprietary_model,
)


@dataclass
class OpenCompassRow:
    """A single model row from an OpenCompass leaderboard."""

    model_name: str
    provider: Optional[str]
    overall_score: Optional[float]
    benchmarks: Dict[str, float]
    rank: Optional[int]
    submission_date: Optional[str]
    source: str

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "overall_score": self.overall_score,
            "benchmarks": self.benchmarks,
            "rank": self.rank,
            "submission_date": self.submission_date,
            "source": self.source,
        }


class OpenCompassScraper:
    """Scrapes both OpenCompass LLM leaderboards using Selenium."""

    GENERAL_URL_TEMPLATE = "https://rank.opencompass.org.cn/leaderboard-llm/?m={month}"
    ACADEMIC_URL_TEMPLATE = "https://rank.opencompass.org.cn/leaderboard-llm-academic/?m={month}"

    BENCHMARK_HEADER_PATTERNS = [
        "C-Eval", "MMLU", "GSM8K", "MATH", "BBH", "HumanEval", "MBPP",
        "CMMLU", "ARC", "DROP", "HellaSwag", "PIQA", "COPA", "WSC",
        "AGIEval", "WiC", "CHID", "AFQMC", "Flores", "TyDiQA",
        "CommonSenseQA", "TriviaQA", "BoolQ", "NATURAL QUESTIONS",
        "Overall", "Average", "Score",
    ]

    def __init__(self, headless: bool = True):
        self.options = Options()
        if headless:
            self.options.add_argument("--headless=new")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )

        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=self.options)
        except Exception:
            logger.warning("ChromeDriverManager failed, falling back to system driver")
            self.driver = webdriver.Chrome(options=self.options)

        self.driver.implicitly_wait(SELENIUM_CONFIG["implicit_wait"])

    def _wait_for_table(self, timeout: int = 30) -> bool:
        """Wait for the leaderboard table to hydrate."""
        selectors = [
            (By.CSS_SELECTOR, "table tbody tr"),
            (By.CSS_SELECTOR, "[class*='table'] tbody tr"),
            (By.CSS_SELECTOR, "[class*='leaderboard'] tbody tr"),
            (By.CSS_SELECTOR, "div[class*='row'] div[class*='model']"),
        ]

        for selector in selectors:
            try:
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located(selector)
                )
                time.sleep(2)
                return True
            except TimeoutException:
                continue

        time.sleep(5)
        for selector in selectors:
            try:
                if selector[1].startswith("table"):
                    if len(self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")) > 5:
                        return True
                else:
                    if len(self.driver.find_elements(*selector)) > 5:
                        return True
            except Exception:
                continue

        return False

    def _get_text(self, elem, xpath: str) -> Optional[str]:
        """Safely extract text from an element."""
        try:
            return elem.find_element(By.XPATH, xpath).text.strip()
        except Exception:
            return None

    def _try_find_cell(self, row, selectors: List[tuple]) -> Optional[object]:
        """Try multiple selectors to find a cell element."""
        for by, sel in selectors:
            try:
                return row.find_element(by, sel)
            except NoSuchElementException:
                continue
        return None

    def _extract_benchmarks_from_header(self, header_cells: List) -> Dict[str, int]:
        """Map benchmark names to column indices from table headers."""
        mapping = {}
        for i, cell in enumerate(header_cells):
            text = cell.text.strip().lower()
            for pattern in self.BENCHMARK_HEADER_PATTERNS:
                if pattern.lower() in text:
                    mapping[pattern] = i
                    break
        return mapping

    def _parse_table(
        self, source: str, min_cols: int = 4
    ) -> List[OpenCompassRow]:
        """Parse the current leaderboard table."""
        try:
            table_selectors = [
                (By.TAG_NAME, "table"),
                (By.CSS_SELECTOR, "[class*='table']"),
            ]

            table = None
            for sel in table_selectors:
                try:
                    table = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located(sel)
                    )
                    break
                except TimeoutException:
                    continue

            if table is None:
                return []

            thead = table.find_element(By.TAG_NAME, "thead")
            header_cells = thead.find_elements(By.TAG_NAME, "th")
            bench_mapping = self._extract_benchmarks_from_header(header_cells)

            tbody = table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")

            results = []
            rank = 0

            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < min_cols:
                    continue

                model_name = cells[0].text.strip()
                if not model_name or len(model_name) < 2:
                    continue

                if is_proprietary_model(model_name):
                    continue

                provider = None
                if len(cells) > 1:
                    provider = cells[1].text.strip() or None
                    if provider and is_proprietary_model("", provider):
                        continue

                overall_score = None
                if len(cells) > 2:
                    try:
                        txt = cells[2].text.strip().replace("%", "").replace(",", "")
                        if txt:
                            overall_score = float(txt)
                    except ValueError:
                        pass

                benchmarks = {}
                for bench_name, col_idx in bench_mapping.items():
                    if col_idx < len(cells):
                        try:
                            txt = cells[col_idx].text.strip().replace("%", "").replace(",", "")
                            if txt:
                                benchmarks[bench_name] = float(txt)
                        except ValueError:
                            pass

                rank += 1

                results.append(OpenCompassRow(
                    model_name=model_name,
                    provider=provider,
                    overall_score=overall_score,
                    benchmarks=benchmarks,
                    rank=rank,
                    submission_date=None,
                    source=source,
                ))

            return results

        except Exception as e:
            logger.error(f"Table parse error: {e}")
            return []

    def _try_alternative_parse(
        self, source: str
    ) -> List[OpenCompassRow]:
        """Fallback: parse using div-based row structure."""
        try:
            results = []
            rank = 0

            row_selectors = [
                "[class*='leaderboard'] [class*='row']",
                "[class*='model-list'] [class*='item']",
                "div[class*='list'] > div",
            ]

            rows = []
            for sel in row_selectors:
                rows = self.driver.find_elements(By.CSS_SELECTOR, sel)
                if rows:
                    break

            for row in rows[:200]:
                try:
                    text = row.text.strip()
                    if not text or len(text) < 10:
                        continue

                    lines = text.split("\n")
                    if len(lines) < 2:
                        continue

                    model_name = lines[0].strip()
                    if is_proprietary_model(model_name):
                        continue

                    provider = None
                    overall_score = None
                    benchmarks = {}

                    if len(lines) > 1:
                        for line in lines[1:]:
                            try:
                                val = float(line.replace("%", "").replace(",", ""))
                                if overall_score is None:
                                    overall_score = val
                                else:
                                    benchmarks[f"score_{len(benchmarks)}"] = val
                            except ValueError:
                                if "Provider" in line or "Source" in line:
                                    provider = line.split(":")[-1].strip()

                    rank += 1
                    results.append(OpenCompassRow(
                        model_name=model_name,
                        provider=provider,
                        overall_score=overall_score,
                        benchmarks=benchmarks,
                        rank=rank,
                        submission_date=None,
                        source=source,
                    ))

                except Exception:
                    continue

            return results

        except Exception as e:
            logger.error(f"Alternative parse failed: {e}")
            return []

    def _click_pagination(self) -> bool:
        """Click pagination or load-more button if present."""
        selectors = [
            (By.XPATH, "//button[contains(text(), 'Load More')]"),
            (By.XPATH, "//button[contains(text(), 'Show More')]"),
            (By.XPATH, "//button[contains(text(), 'Next')]"),
            (By.CSS_SELECTOR, "[class*='pagination'] button"),
            (By.CSS_SELECTOR, "[data-testid*='load-more']"),
        ]

        for by, selector in selectors:
            try:
                btn = self.driver.find_element(by, selector)
                if btn.is_displayed() and btn.is_enabled():
                    btn.click()
                    time.sleep(2)
                    return True
            except NoSuchElementException:
                continue
        return False

    def scrape(self, source: str, month: str = "REALTIME") -> List[Dict]:
        """Main scraping method with pagination for a single leaderboard."""
        if source == "general":
            url = self.GENERAL_URL_TEMPLATE.format(month=month)
        else:
            url = self.ACADEMIC_URL_TEMPLATE.format(month=month)

        logger.info(f"Scraping OpenCompass {source}: {url}")

        try:
            self.driver.get(url)

            if not self._wait_for_table():
                logger.error(f"Table hydration failed for {source}")
                return []

            all_rows: List[OpenCompassRow] = []
            seen_names: set = set()
            iteration = 0

            while iteration < 10:
                rows = self._parse_table(source)
                if not rows:
                    rows = self._try_alternative_parse(source)
                    if not rows:
                        break

                new_this_iter = 0
                for row in rows:
                    if row.model_name not in seen_names:
                        seen_names.add(row.model_name)
                        all_rows.append(row)
                        new_this_iter += 1

                logger.info(f"{source}: total unique models={len(all_rows)} (+{new_this_iter} new)")

                if new_this_iter == 0:
                    break
                if not self._click_pagination():
                    break

                iteration += 1

            logger.success(f"Scraped {len(all_rows)} models from {source}")

            return [r.to_dict() for r in all_rows]

        except Exception as e:
            logger.error(f"Scraping failed for {source}: {e}")
            return []

    def scrape_both(
        self,
        general_month: str = "26-04",
        academic_month: str = "REALTIME",
    ) -> tuple:
        """Scrape both leaderboards in sequence."""
        general = self.scrape("general", general_month)
        academic = self.scrape("academic", academic_month)
        return general, academic

    def save_temp(self, data: List[Dict], dest: Path) -> None:
        """Save scraped data to a temp file."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} records to {dest}")

    def close(self):
        """Close the browser driver."""
        try:
            self.driver.quit()
        except Exception:
            pass