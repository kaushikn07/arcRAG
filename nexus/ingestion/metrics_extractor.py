"""
Financial Metrics Extractor for NEXUS

Extracts structured financial metrics from document text using LLM.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger
import re

import httpx
from config import settings


@dataclass
class FinancialMetric:
    """Represents an extracted financial metric."""
    metric_name: str
    value: float
    unit: str
    period: Optional[str] = None
    company: Optional[str] = None
    year: Optional[int] = None
    source_section_id: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MetricsExtractor:
    """
    Extract financial metrics from document text using LLM.
    
    Identifies and structures key financial figures like:
    - Revenue, Net Income, EBITDA
    - Assets, Liabilities, Equity
    - Cash Flow metrics
    - Ratios (P/E, Debt/Equity, etc.)
    """

    def __init__(self, model: str = None):
        """
        Initialize the extractor.
        
        Args:
            model: LLM model to use for extraction
        """
        self.model = model or settings.llm_model_classify
        self.base_url = settings.openrouter_base_url
        self.api_key = settings.openrouter_api_key

    async def extract_metrics(
        self,
        text: str,
        company: Optional[str] = None,
        year: Optional[int] = None,
        section_id: Optional[int] = None
    ) -> List[FinancialMetric]:
        """
        Extract financial metrics from text.
        
        Args:
            text: Text to extract from
            company: Company name
            year: Fiscal year
            section_id: Source section ID
            
        Returns:
            List of FinancialMetric objects
        """
        if not text.strip():
            return []

        prompt = self._build_prompt(text, company, year)
        
        try:
            response_text = await self._call_llm(prompt)
            metrics = self._parse_metrics(response_text, company, year, section_id)
            
            logger.info(f"Extracted {len(metrics)} financial metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            return []

    def _build_prompt(
        self,
        text: str,
        company: Optional[str],
        year: Optional[int]
    ) -> str:
        """Build the extraction prompt."""
        # Truncate if too long
        max_chars = 4000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        context = ""
        if company:
            context += f"Company: {company}\n"
        if year:
            context += f"Fiscal Year: {year}\n"
        
        prompt = f"""You are an expert financial analyst. Extract ALL financial metrics from the following text.

{context}
Focus on these metric types:
- Revenue, Net Income, Operating Income, EBITDA, EBIT
- Total Assets, Current Assets, Non-current Assets
- Total Liabilities, Current Liabilities, Long-term Debt
- Shareholders' Equity, Retained Earnings
- Cash Flow from Operations, Investing, Financing
- EPS, P/E Ratio, Debt-to-Equity, Current Ratio
- Any other numerical financial figures with units

Text:
{text}

Output format (JSON array):
[
    {{
        "metric_name": "Revenue",
        "value": 15000000000,
        "unit": "USD",
        "period": "FY2023"
    }},
    ...
]

Only output valid JSON. If no metrics found, output [].

Metrics:"""
        
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/nexus-rag",
            "X-Title": "NEXUS"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise financial data extractor. Output only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        url = f"{self.base_url}/chat/completions"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def _parse_metrics(
        self,
        response_text: str,
        company: Optional[str],
        year: Optional[int],
        section_id: Optional[int]
    ) -> List[FinancialMetric]:
        """Parse metrics from LLM response."""
        import json
        
        metrics = []
        
        # Try to extract JSON
        try:
            # Find JSON array in response
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                
                for item in data:
                    if isinstance(item, dict):
                        metric = FinancialMetric(
                            metric_name=item.get('metric_name', 'Unknown'),
                            value=float(item.get('value', 0)),
                            unit=item.get('unit', 'USD'),
                            period=item.get('period'),
                            company=company,
                            year=year,
                            source_section_id=section_id,
                            metadata={'raw_response': response_text}
                        )
                        metrics.append(metric)
                        
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
            
            # Fallback: try regex patterns
            metrics = self._regex_extract(response_text, company, year, section_id)
        
        return metrics

    def _regex_extract(
        self,
        text: str,
        company: Optional[str],
        year: Optional[int],
        section_id: Optional[int]
    ) -> List[FinancialMetric]:
        """Fallback regex-based extraction."""
        metrics = []
        
        # Common financial metric patterns
        patterns = [
            (r'(?:revenue|sales)[^\d]*([\d,\.]+)\s*(million|billion|USD|\$)?', 'Revenue'),
            (r'(?:net income|net profit)[^\d]*([\d,\.]+)\s*(million|billion|USD|\$)?', 'Net Income'),
            (r'(?:EBITDA)[^\d]*([\d,\.]+)\s*(million|billion|USD|\$)?', 'EBITDA'),
            (r'(?:total assets)[^\d]*([\d,\.]+)\s*(million|billion|USD|\$)?', 'Total Assets'),
            (r'(?:total liabilities)[^\d]*([\d,\.]+)\s*(million|billion|USD|\$)?', 'Total Liabilities'),
        ]
        
        for pattern, default_name in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value_str, unit_str = match
                    
                    # Parse value
                    value = float(value_str.replace(',', ''))
                    
                    # Parse unit
                    unit = 'USD'
                    multiplier = 1
                    if unit_str:
                        unit_str = unit_str.lower()
                        if 'million' in unit_str:
                            multiplier = 1_000_000
                        elif 'billion' in unit_str:
                            multiplier = 1_000_000_000
                        if '$' in unit_str or 'usd' in unit_str:
                            unit = 'USD'
                    
                    value *= multiplier
                    
                    metrics.append(FinancialMetric(
                        metric_name=default_name,
                        value=value,
                        unit=unit,
                        company=company,
                        year=year,
                        source_section_id=section_id
                    ))
                    
                except (ValueError, TypeError):
                    continue
        
        return metrics


# Global extractor instance
_extractor: Optional[MetricsExtractor] = None


def get_extractor() -> MetricsExtractor:
    """Get the global metrics extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = MetricsExtractor()
    return _extractor


async def extract_financial_metrics(
    text: str,
    company: Optional[str] = None,
    year: Optional[int] = None,
    **kwargs
) -> List[FinancialMetric]:
    """
    Convenience function to extract financial metrics.
    
    Args:
        text: Text to extract from
        company: Company name
        year: Fiscal year
        **kwargs: Additional arguments
        
    Returns:
        List of FinancialMetric objects
    """
    extractor = get_extractor()
    return await extractor.extract_metrics(text, company, year, **kwargs)
