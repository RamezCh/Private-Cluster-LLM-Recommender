"""Web scraping module for Artificial Analysis using Selenium."""

import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from loguru import logger

from src.config import TEMP_PERFORMANCE_FILE


@dataclass
class PerformanceData:
    """Model performance data from Artificial Analysis."""

    model_name: str
    intelligence_index: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class WebScraper:
    """Scrapes performance metrics from artificialanalysis.ai using Selenium."""

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

        # Use webdriver-manager to automatically handle driver versioning
        # Note: In Docker/Linux environments with 'chromium' package,
        # we should use ChromeDriverManager().install() or rely on the system driver
        # For maximum robustness, we use the service with ChromeDriverManager
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=self.options)
        except Exception as e:
            logger.warning(
                f"ChromeDriverManager failed ({e}), falling back to system driver"
            )
            self.driver = webdriver.Chrome(options=self.options)

        self.driver.implicitly_wait(10)

        self.url = "https://artificialanalysis.ai/leaderboards/models"
        self.data: List[PerformanceData] = []

    def _wait_for_hydration(self, timeout: int = 30) -> bool:
        """Wait for JS-rendered table to fully load."""
        selectors = [
            (By.CSS_SELECTOR, "table tbody tr"),
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
        return len(self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")) > 5

    def _get_text(self, row, xpath: str) -> Optional[str]:
        """Safely extract text from element."""
        try:
            el = row.find_element(By.XPATH, xpath)
            return el.text.strip()
        except Exception:
            return None

    def _parse_row(self, row) -> Optional[PerformanceData]:
        """Parse a single table row."""
        model_name = self._get_text(row, ".//td[1]")
        if not model_name or len(model_name) < 3:
            return None

        intelligence = None
        text = self._get_text(row, ".//td[4]")
        if text:
            try:
                intelligence = float(text.replace(",", ""))
            except ValueError:
                pass

        throughput = None
        text = self._get_text(row, ".//td[6]")
        if text:
            try:
                throughput = float(
                    text.replace(",", "").replace("tok/s", "").replace("tokens/s", "")
                )
            except ValueError:
                pass

        return PerformanceData(
            model_name=model_name,
            intelligence_index=intelligence,
            throughput_tokens_per_sec=throughput,
        )

    def _parse_table(self) -> List[PerformanceData]:
        """Parse the rendered table."""
        try:
            table = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            rows = table.find_element(By.TAG_NAME, "tbody").find_elements(
                By.TAG_NAME, "tr"
            )
            return [p for row in rows if (p := self._parse_row(row))]
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return []

    def _click_load_more(self) -> bool:
        """Click pagination button if present."""
        selectors = [
            (By.XPATH, "//button[contains(text(), 'Load More')]"),
            (By.XPATH, "//button[contains(text(), 'Show More')]"),
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

    def scrape(self, save_temp: bool = True) -> List[Dict]:
        """Main scraping method with pagination."""
        logger.info(f"Scraping {self.url}")

        try:
            self.driver.get(self.url)

            if not self._wait_for_hydration():
                logger.error("Page hydration failed")
                return []

            all_data: List[PerformanceData] = []
            seen_names: set = set()
            iteration = 0

            while iteration < 20:
                current = self._parse_table()

                # Only add models not seen before (full table is returned each time)
                new_this_iter = 0
                for item in current:
                    if item.model_name not in seen_names:
                        seen_names.add(item.model_name)
                        all_data.append(item)
                        new_this_iter += 1

                logger.info(f"Total unique models: {len(all_data)} (+{new_this_iter} new)")

                # Stop if no new models found or no Load More button
                if new_this_iter == 0 or not self._click_load_more():
                    break

                iteration += 1

            self.data = all_data

            if save_temp:
                TEMP_PERFORMANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(TEMP_PERFORMANCE_FILE, "w", encoding="utf-8") as f:
                    json.dump([d.to_dict() for d in all_data], f, indent=2)
                logger.info(f"Saved to {TEMP_PERFORMANCE_FILE}")

            return [d.to_dict() for d in all_data]

        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return []
        finally:
            try:
                self.driver.quit()
            except Exception:
                pass
