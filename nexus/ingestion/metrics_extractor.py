"""Financial metrics extractor using LLM."""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Optional

import httpx
from loguru import logger

from config import settings
from ingestion.parser import ParsedSection


@dataclass
class ExtractedMetric:
    """Represents an extracted financial metric."""

    metric_name: str
    value: float
    unit: str
    period: str  # annual/Q1/Q2/Q3/Q4
    doc_id: str
    company: str
    year: int
    source_section_id: Optional[str] = None


class FinancialMetricsExtractor:
    """Extract financial metrics from document sections using LLM."""

    VALID_METRICS = {
        "revenue",
        "net_income",
        "eps",
        "operating_income",
        "gross_profit",
        "total_assets",
        "total_debt",
    }

    def __init__(self):
        self.base_url = settings.openrouter_base_url.rstrip("/")
        self.api_key = settings.openrouter_api_key
        self.model = settings.llm_model_fast
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/nexus-rag",
            "Content-Type": "application/json",
        }
        # Rate limit: max 10 concurrent requests
        self.semaphore = asyncio.Semaphore(10)

    async def _call_llm(self, prompt: str) -> str:
        """Call OpenRouter API with the given prompt."""
        async with self.semaphore:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                }
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error from OpenRouter: {e.response.status_code} - {e.response.text}")
                    raise
                except Exception as e:
                    logger.error(f"Error calling OpenRouter: {e}")
                    raise

    async def extract_metrics(
        self, section: ParsedSection, doc_id: str, company: str, year: int
    ) -> list[dict]:
        """
        Extract financial metrics from a section.

        Only processes sections with content_type in ('table', 'mixed').

        Args:
            section: The ParsedSection to extract from
            doc_id: Document ID
            company: Company name
            year: Fiscal year

        Returns:
            List of dicts with keys: metric_name, value, unit, period
        """
        # Only process table or mixed content
        if section.content_type not in ("table", "mixed"):
            return []

        # Get text content
        text = section.raw_text
        if section.table_data and section.table_data.get("markdown"):
            text = f"{section.raw_text}\n\nTable:\n{section.table_data['markdown']}"

        if not text.strip():
            return []

        prompt = (
            "Extract financial metrics from this text. Return a JSON array of objects with keys:\n"
            "metric_name, value (number only, in millions USD), unit, period (annual/Q1/Q2/Q3/Q4).\n"
            "Only include: revenue, net_income, eps, operating_income, gross_profit, total_assets, total_debt.\n"
            "Return [] if no financial metrics found. Return ONLY the JSON array.\n\n"
            f"Company: {company}\nYear: {year}\n\nText:\n{text}"
        )

        try:
            response_text = await self._call_llm(prompt)

            # Parse JSON array from response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            metrics_data = json.loads(response_text)

            if not isinstance(metrics_data, list):
                logger.warning(f"Expected JSON array, got {type(metrics_data)}")
                return []

            result = []
            for item in metrics_data:
                if not isinstance(item, dict):
                    continue

                metric_name = item.get("metric_name", "").lower().replace(" ", "_").replace("-", "_")

                # Filter to valid metrics only
                if metric_name not in self.VALID_METRICS:
                    continue

                # Parse value - strip $, B, M, commas
                value_raw = item.get("value")
                if value_raw is None:
                    continue

                value = self._parse_numeric_value(value_raw)
                if value is None:
                    continue

                unit = item.get("unit", "USD")
                period = item.get("period", "annual")

                # Normalize period
                period_lower = period.lower()
                if period_lower in ("annual", "fy", "full year", "year"):
                    period = "annual"
                elif period_lower in ("q1", "quarter 1", "first quarter"):
                    period = "Q1"
                elif period_lower in ("q2", "quarter 2", "second quarter"):
                    period = "Q2"
                elif period_lower in ("q3", "quarter 3", "third quarter"):
                    period = "Q3"
                elif period_lower in ("q4", "quarter 4", "fourth quarter"):
                    period = "Q4"
                else:
                    period = "annual"

                result.append(
                    {
                        "metric_name": metric_name,
                        "value": value,
                        "unit": unit,
                        "period": period,
                        "doc_id": doc_id,
                        "company": company,
                        "year": year,
                        "source_section_id": section.section_id,
                    }
                )

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}. Response: {response_text[:200]}")
            return []
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            return []

    def _parse_numeric_value(self, value_raw) -> Optional[float]:
        """
        Parse a numeric value, stripping $, B, M, commas.

        Args:
            value_raw: Raw value (string or number)

        Returns:
            Float value in millions USD, or None if parsing fails
        """
        if isinstance(value_raw, (int, float)):
            return float(value_raw)

        if not isinstance(value_raw, str):
            return None

        # Strip whitespace
        value_str = value_raw.strip()

        # Check for billions/millions suffix
        multiplier = 1.0
        value_lower = value_str.lower()
        if "billion" in value_lower or "b" in value_lower:
            multiplier = 1000.0  # Convert to millions
        elif "million" in value_lower or "m" in value_lower:
            multiplier = 1.0  # Already in millions

        # Remove non-numeric characters except digits, dots, minus, commas
        cleaned = re.sub(r"[^\d.\-,]", "", value_str)
        cleaned = cleaned.replace(",", "")

        try:
            value = float(cleaned) * multiplier
            return value
        except (ValueError, TypeError):
            return None

    async def extract_and_store_all(
        self, sections: list[ParsedSection], doc_id: str, company: str, year: int
    ) -> int:
        """
        Extract metrics from all sections and store in database.

        Args:
            sections: List of ParsedSection objects
            doc_id: Document ID
            company: Company name
            year: Fiscal year

        Returns:
            Total number of metrics stored
        """
        from db.neon import execute

        total_stored = 0

        # Process sections
        tasks = []
        for section in sections:
            task = self.extract_metrics(section, doc_id, company, year)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Store results
        for section, result in zip(sections, results):
            if isinstance(result, Exception):
                logger.error(f"Error processing section {section.section_id}: {result}")
                continue

            metrics = result
            for metric in metrics:
                insert_query = """
                INSERT INTO financial_metrics 
                (metric_id, doc_id, company, year, period, metric_name, value, unit, source_section_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (metric_id) DO NOTHING
                """
                # Generate deterministic metric_id
                import hashlib

                metric_id = hashlib.sha256(
                    f"{doc_id}:{metric['metric_name']}:{metric['period']}".encode()
                ).hexdigest()[:16]

                try:
                    await execute(
                        insert_query,
                        metric_id,
                        metric["doc_id"],
                        metric["company"],
                        metric["year"],
                        metric["period"],
                        metric["metric_name"],
                        metric["value"],
                        metric["unit"],
                        metric["source_section_id"],
                    )
                    total_stored += 1
                except Exception as e:
                    logger.error(f"Failed to store metric {metric_id}: {e}")

        return total_stored


# Convenience function
async def extract_metrics(section: ParsedSection, doc_id: str, company: str, year: int) -> list[dict]:
    """Extract financial metrics from a section."""
    extractor = FinancialMetricsExtractor()
    return await extractor.extract_metrics(section, doc_id, company, year)
