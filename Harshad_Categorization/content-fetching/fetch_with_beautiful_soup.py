"""
Fetch a URL and extract cleaned page text (title, meta, headings, body) using BeautifulSoup.

Public API: ``fetch_page_content`` and ``PageContent``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag

__all__ = ["PageContent", "fetch_page_content"]

DEFAULT_REQUEST_TIMEOUT = 15


@dataclass
class PageContent:
    url: str
    domain: str
    title: str
    meta_description: str
    headings: List[str]
    body_text: str


def _normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _fetch_url_html(url: str, timeout: int) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _get_meta_content(soup: BeautifulSoup, attrs: Dict[str, str]) -> str:
    tag = soup.find("meta", attrs=attrs)
    if tag and tag.get("content"):
        return _normalize_text(tag.get("content"))
    return ""


def _remove_noise_nodes(soup: BeautifulSoup) -> None:
    selectors = [
        "script", "style", "noscript", "svg", "iframe", "canvas", "form",
        "nav", "footer", "header", "aside"
    ]
    for selector in selectors:
        for tag in soup.select(selector):
            tag.decompose()

    for tag in soup.find_all(True):
        if tag is None or getattr(tag, "attrs", None) is None:
            continue
        classes = " ".join(tag.get("class", [])) if tag.get("class") else ""
        id_ = tag.get("id", "")
        marker = f"{classes} {id_}".lower()
        if any(x in marker for x in [
            "cookie", "consent", "newsletter", "subscribe", "promo",
            "advert", "ad-", "ads", "banner", "breadcrumb", "related",
            "social-share", "share", "outbrain", "taboola", "recommended"
        ]):
            tag.decompose()


def _extract_best_text_container(soup: BeautifulSoup) -> Optional[Tag]:
    priority_selectors = [
        "article",
        "main",
        "[role='main']",
        ".article",
        ".post",
        ".entry-content",
        ".article-content",
        ".post-content",
        ".story-body",
        ".content",
    ]
    for selector in priority_selectors:
        node = soup.select_one(selector)
        if node:
            return node

    body = soup.body
    if not body:
        return None

    best_node = None
    best_score = -1

    for node in body.find_all(["div", "section"], recursive=True):
        text = _normalize_text(node.get_text(" ", strip=True))
        if not text:
            continue

        p_count = len(node.find_all("p"))
        heading_count = len(node.find_all(["h1", "h2", "h3"]))
        text_len = len(text)
        score = text_len + (p_count * 200) + (heading_count * 100)

        if score > best_score:
            best_score = score
            best_node = node

    return best_node or body


def _extract_headings(root: Tag, limit: int = 8) -> List[str]:
    headings = []
    for tag in root.find_all(["h1", "h2", "h3"]):
        txt = _normalize_text(tag.get_text(" ", strip=True))
        if txt and len(txt) > 2:
            headings.append(txt)
    return _dedupe_preserve_order(headings)[:limit]


def _extract_body_text(root: Tag) -> str:
    paras = []
    for p in root.find_all(["p", "li"]):
        txt = _normalize_text(p.get_text(" ", strip=True))
        if txt and len(txt) >= 40:
            paras.append(txt)

    if not paras:
        return _normalize_text(root.get_text(" ", strip=True))

    return "\n".join(_dedupe_preserve_order(paras))


def _strip_noise(text: str) -> str:
    text = _normalize_text(text)
    noise_patterns = [
        r"\bprivacy policy\b",
        r"\bterms\s*(and|&)\s*conditions\b",
        r"\bcontact us\b",
        r"\bfollow us\b",
        r"\bdownload app\b",
        r"\badvertisement\b",
        r"\ball rights reserved\b",
        r"\bnews archive\b",
        r"\btopics archive\b",
        r"\bread more\b",
        r"\bclick here\b",
    ]
    for pat in noise_patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _build_page_content_from_html(html: str, url: str) -> PageContent:
    soup = BeautifulSoup(html, "html.parser")
    _remove_noise_nodes(soup)

    title = (
        _get_meta_content(soup, {"property": "og:title"})
        or _get_meta_content(soup, {"name": "twitter:title"})
        or _normalize_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    )

    meta_description = (
        _get_meta_content(soup, {"property": "og:description"})
        or _get_meta_content(soup, {"name": "description"})
        or _get_meta_content(soup, {"name": "twitter:description"})
    )

    root = _extract_best_text_container(soup)
    if root is None:
        headings = []
        body_text = ""
    else:
        headings = _extract_headings(root)
        body_text = _extract_body_text(root)

    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    return PageContent(
        url=url,
        domain=domain,
        title=_strip_noise(title),
        meta_description=_strip_noise(meta_description),
        headings=[h for h in (_strip_noise(h) for h in headings) if h],
        body_text=_strip_noise(body_text),
    )


def fetch_page_content(url: str, *, timeout: int = DEFAULT_REQUEST_TIMEOUT) -> PageContent:
    """
    GET ``url``, parse HTML with BeautifulSoup, and return structured, cleaned ``PageContent``.

    Raises ``requests.HTTPError`` on non-success HTTP status; ``requests`` exceptions on
    network/timeout errors.
    """
    html = _fetch_url_html(url, timeout)
    return _build_page_content_from_html(html, url)

