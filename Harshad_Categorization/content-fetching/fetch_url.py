#!/usr/bin/env python3
"""
Fetch URL content and extract plain text (no HTML, JavaScript, or images).
Uses only the standard library.
"""

import gzip
import html
import html.parser
import re
import ssl
import time
import urllib.error
import urllib.request
from typing import Optional


class _TextExtractor(html.parser.HTMLParser):
    """Extract visible text from HTML, skipping script, style, and other non-content tags."""

    SKIP_TAGS = frozenset(
        ('script', 'style', 'noscript', 'iframe', 'embed', 'object', 'svg', 'head', 'meta', 'link')
    )

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0 and data:
            self._parts.append(data)

    def get_text(self):
        return ' '.join(self._parts)


def fetch_url_content(url: str, timeout: int = 45) -> Optional[str]:
    """
    Fetch content from a URL and extract text only (no HTML, no JavaScript, no images).
    Uses only the standard library (urllib, html.parser).
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    req = urllib.request.Request(url, headers=headers, method='GET')
    ctx_verified = ssl.create_default_context()
    ctx_unverified = ssl.create_default_context()
    ctx_unverified.check_hostname = False
    ctx_unverified.verify_mode = ssl.CERT_NONE

    for attempt in range(3):
        try:
            try:
                resp = urllib.request.urlopen(req, timeout=timeout, context=ctx_verified)
            except urllib.error.URLError as e:
                # urlopen wraps SSL errors in URLError; retry with unverified context if certs fail
                if isinstance(getattr(e, 'reason', None), ssl.SSLError):
                    resp = urllib.request.urlopen(req, timeout=timeout, context=ctx_unverified)
                else:
                    raise

            with resp:
                raw = resp.read()
                content_type = (resp.headers.get('Content-Type') or '').lower()
                encoding = resp.headers.get_content_charset() or 'utf-8'
                enc = resp.headers.get('Content-Encoding', '')
                if 'gzip' in enc.lower():
                    try:
                        raw = gzip.decompress(raw)
                    except OSError:
                        pass
                try:
                    html_str = raw.decode(encoding)
                except (UnicodeDecodeError, LookupError):
                    html_str = raw.decode('utf-8', errors='replace')

                if 'html' not in content_type:
                    text = html_str[:50000]
                    if len(html_str) > 50000:
                        text += " ... [content truncated]"
                    return text.strip()

                parser = _TextExtractor()
                try:
                    parser.feed(html_str)
                except Exception:
                    pass
                text = parser.get_text()
                # Fallback: if parser got no text (e.g. malformed HTML), strip tags with regex
                if not text or len(text) < 100:
                    no_script = re.sub(
                        r'<script[^>]*>.*?</script>',
                        ' ',
                        html_str,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    no_style = re.sub(
                        r'<style[^>]*>.*?</style>',
                        ' ',
                        no_script,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    text = re.sub(r'<[^>]+>', ' ', no_style)
                text = html.unescape(text)
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > 50000:
                    text = text[:50000] + " ... [content truncated]"
                return text if text else None

        except urllib.error.HTTPError as e:
            print(f"Error fetching URL {url}: HTTP {e.code} {e.reason}")
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            return None
        except urllib.error.URLError as e:
            print(f"Error fetching URL {url}: {e.reason}")
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            return None
        except OSError as e:
            print(f"Error fetching URL {url}: {str(e)}")
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            return None
        except Exception as e:
            print(f"Error processing content from {url}: {str(e)}")
            return None

    return None

