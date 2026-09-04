"""Resilient, official-NICE-only HTML extraction helpers."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

NICE_BASE_URL = "https://www.nice.org.uk/guidance/"
_TA_ID = re.compile(r"^[A-Za-z]+\d+$")


def _guidance_url(ta_id: str, suffix: str = "") -> str:
    normalized = str(ta_id).strip().lower()
    if not _TA_ID.fullmatch(normalized):
        raise ValueError("ta_id must look like TA875 or HST31")
    return f"{NICE_BASE_URL}{normalized}{suffix}"


def _fetch_text(url: str, session: requests.Session | None = None) -> str:
    client = session or requests.Session()
    try:
        response = client.get(url, timeout=20, headers={"User-Agent": "NICE-Appraisal-Prediction/1.0"})
        response.raise_for_status()
    except requests.RequestException:
        return ""
    soup = BeautifulSoup(response.text, "lxml")
    main = soup.find("main") or soup.body
    if not main:
        return ""
    for element in main.select("script, style, nav, footer, noscript"):
        element.decompose()
    return " ".join(main.stripped_strings)


def scrape_recommendations(ta_id: str) -> str:
    """Return official NICE recommendation text, or an empty string when unavailable."""
    return _fetch_text(_guidance_url(ta_id, "/chapter/1-Recommendations"))


def scrape_history(ta_id: str) -> str:
    """Return official NICE history text, or an empty string when unavailable."""
    return _fetch_text(_guidance_url(ta_id, "/history"))


def get_document_links(ta_id: str) -> list[str]:
    """Return deduplicated official NICE resource/PDF URLs without downloading them."""
    url = _guidance_url(ta_id)
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "NICE-Appraisal-Prediction/1.0"})
        response.raise_for_status()
    except requests.RequestException:
        return []
    soup = BeautifulSoup(response.text, "lxml")
    links: list[str] = []
    for anchor in soup.select("a[href]"):
        href = urljoin(url, anchor["href"])
        parsed = urlparse(href)
        label = anchor.get_text(" ", strip=True).lower()
        if parsed.netloc.endswith("nice.org.uk") and (href.lower().endswith(".pdf") or "resource" in href or "document" in label):
            if href not in links:
                links.append(href)
    return links


def scrape_guidance(ta_id: str) -> dict[str, Any]:
    """Collect non-fatal guidance content for ingestion by the RAG builder."""
    normalized = str(ta_id).upper()
    return {
        "ta_id": normalized,
        "recommendations": scrape_recommendations(normalized),
        "history": scrape_history(normalized),
        "document_links": get_document_links(normalized),
    }
