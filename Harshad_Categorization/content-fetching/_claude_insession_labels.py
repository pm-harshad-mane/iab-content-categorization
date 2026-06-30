from __future__ import annotations
import csv, json, re
from pathlib import Path

SRC = Path("02_fetched_url_content_files/new_urls.jsonl")
TAX = Path("taxonomy/Content_Taxonomy_3.1_6.tsv")
OUT = Path("09_fetched_url_content_categories_by_Claude/new_urls__claude-opus-4-8.jsonl")
MODEL = "claude-opus-4-8"
MAX_BODY = 6000

# Labels assigned in-session by Claude Code (claude-opus-4-8), keyed by 1-based line index.
# Each value is a ranked list of (unique_id, score). Record 4 is a source error (no labels).
LABELS = {
    1:  [("80DV8O",0.78),("628",0.62),("635",0.5),("602",0.45),("596",0.4)],
    2:  [("386",0.85),("383",0.78),("380",0.6),("388",0.5),("387",0.35)],
    3:  [("386",0.85),("383",0.6),("388",0.5),("380",0.45),("387",0.3)],
    5:  [("387",0.88),("386",0.8),("388",0.55),("8YPBBL",0.4)],
    6:  [("387",0.87),("386",0.8),("388",0.55),("8YPBBL",0.4)],
    7:  [("386",0.8),("388",0.6),("389",0.55),("I4GWl6",0.4)],
    8:  [("386",0.72),("388",0.6),("Z7rJBM",0.45),("122",0.4)],
    9:  [("472",0.92),("464",0.72),("118",0.35),("388",0.3)],
    10: [("597",0.8),("386",0.65),("115",0.5),("122",0.45),("388",0.4)],
    11: [("389",0.9),("I4GWl6",0.65),("386",0.45),("380",0.35)],
    12: [("386",0.7),("388",0.65),("97",0.4),("286",0.3)],
    13: [("383",0.7),("Z7rJBM",0.6),("380",0.55),("386",0.4)],
    14: [("386",0.85),("388",0.65),("80",0.35),("389",0.3)],
    15: [("389",0.92),("I4GWl6",0.55),("386",0.4)],
    16: [("386",0.72),("388",0.6),("389",0.5)],
    17: [("383",0.7),("380",0.6),("386",0.55),("432",0.3)],
    18: [("386",0.78),("388",0.6),("110",0.5),("467",0.3)],
    19: [("380",0.7),("386",0.5),("383",0.45),("I4GWl6",0.4)],
    20: [("597",0.78),("386",0.7),("388",0.5),("115",0.45)],
    21: [("386",0.78),("388",0.58),("389",0.45)],
    22: [("640",0.72),("JLBCU7",0.6),("A0AH3G",0.35)],
    23: [("597",0.78),("386",0.68),("122",0.45),("388",0.45)],
    24: [("68",0.65),("106",0.6),("89",0.45),("93",0.4),("53",0.4)],
    25: [("413",0.7),("80",0.62),("410",0.5),("95",0.4),("52",0.4)],
    26: [("413",0.6),("95",0.55),("597",0.55),("80",0.45),("410",0.4)],
    27: [("81",0.78),("80",0.55),("386",0.4),("413",0.3)],
    28: [("95",0.6),("413",0.55),("80",0.45),("410",0.4),("69",0.35)],
    29: [("597",0.72),("386",0.68),("122",0.5),("115",0.4)],
    30: [("597",0.72),("383",0.5),("115",0.5),("628",0.35)],
    31: [("597",0.78),("122",0.55),("386",0.55),("388",0.45)],
    32: [("68",0.75),("106",0.6),("93",0.5),("53",0.4),("640",0.35)],
    33: [("597",0.78),("386",0.68),("388",0.45),("122",0.4)],
    34: [("635",0.82),("632",0.6),("596",0.4),("432",0.3)],
    35: [("29",0.78),("474",0.5),("22",0.4),("481",0.4),("632",0.3)],
    36: [("597",0.78),("386",0.7),("388",0.45),("122",0.4)],
    37: [("634",0.72),("474",0.5),("632",0.45),("640",0.4),("481",0.35)],
    38: [("471",0.85),("464",0.65)],
    39: [("386",0.62),("122",0.5),("388",0.45),("389",0.4)],
    40: [("324",0.85),("652",0.55),("JLBCU7",0.55),("643",0.3)],
    41: [("597",0.78),("386",0.7),("388",0.45),("122",0.4)],
    42: [("597",0.85),("115",0.55),("596",0.45)],
    43: [("269",0.85),("698",0.45),("270",0.4),("683",0.35)],
    44: [("42",0.8),("48",0.6),("636",0.45),("326",0.4),("474",0.35)],
    45: [("608",0.55),("619",0.5),("602",0.5),("474",0.45),("618",0.4)],
    46: [("472",0.92),("464",0.7),("118",0.35),("388",0.3)],
    47: [("545",0.9),("483",0.7),("490",0.3)],
    48: [("545",0.9),("483",0.7)],
    49: [("484",0.78),("483",0.55),("380",0.45),("383",0.4)],
    50: [("491",0.9),("483",0.7)],
    51: [("533",0.9),("483",0.7)],
    52: [("491",0.9),("483",0.7)],
    53: [("324",0.78),("JLBCU7",0.6),("155",0.4),("93",0.4)],
    54: [("433",0.6),("338",0.45),("JLBCU7",0.45),("371",0.35),("187",0.3)],
    55: [("640",0.82),("JLBCU7",0.55),("647",0.5),("331",0.35)],
    56: [("472",0.92),("464",0.68)],
    57: [("472",0.92),("464",0.68),("471",0.35)],
    58: [("465",0.8),("464",0.68),("468",0.3)],
    59: [("465",0.65),("464",0.6),("297",0.45),("468",0.4),("286",0.4)],
    60: [("464",0.55),("680",0.5),("465",0.4),("683",0.3),("432",0.3)],
    61: [("42",0.7),("464",0.5),("313",0.35),("EZWB7V",0.35)],
    62: [("381",0.7),("386",0.6),("388",0.5),("394",0.4)],
    63: [("380",0.65),("383",0.6),("386",0.55),("388",0.4)],
    64: [("386",0.6),("210",0.5),("388",0.45),("229",0.35),("80",0.35)],
}


def norm(v):
    return re.sub(r"\s+", " ", ("" if v is None else str(v)).strip())


def load_tax(path):
    by_id = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            uid = norm(row.get("Unique ID"))
            if uid:
                by_id[uid] = {"unique_id": uid, "parent_id": norm(row.get("Parent")), "path": norm(row.get("Path"))}
    return by_id


def page_text(r):
    parts = []
    url = norm(r.get("url") or r.get("input_url")); dom = norm(r.get("domain"))
    title = norm(r.get("title")); desc = norm(r.get("meta_description"))
    heads = [norm(x) for x in (r.get("headings") or []) if norm(x)]
    body = norm(r.get("body_text"))
    if url: parts.append(f"url: {url}")
    if dom: parts.append(f"domain: {dom}")
    if title: parts.append(f"title: {title}")
    if desc: parts.append(f"description: {desc}")
    if heads: parts.append("headings: " + " | ".join(heads[:8]))
    if body: parts.append(f"content: {body[:MAX_BODY]}")
    return " || ".join(parts)


def score_gap(cats):
    if not cats: return None
    t1 = cats[0]["score"]
    if len(cats) == 1: return float(t1)
    t2 = cats[1]["score"]
    return round(float(t1) - float(t2), 6)


def bucket(cats):
    if not cats: return "discard"
    g = score_gap(cats); n = len(cats)
    if g is None: return "medium"
    if n >= 5 and g >= 0.15: return "high"
    if n >= 3 and g >= 0.08: return "medium"
    return "low"


def model_details():
    return {
        "provider": "anthropic",
        "cloud_model": MODEL,
        "method": "claude_code_in_session",
        "endpoint": None,
        "taxonomy_source": str(TAX),
        "taxonomy_prompt_fields": ["Unique ID", "Path", "Description", "Keywords"],
        "top_k": 5,
        "max_body_chars": MAX_BODY,
        "prompt_layout": "taxonomy_first_then_page_content",
        "note": "Labels assigned by Claude Code (claude-opus-4-8) within an interactive session; no Messages API call was made, so no token usage is recorded.",
    }


def main():
    by_id = load_tax(TAX)
    md = model_details()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    missing_ids = set()
    ok = err = 0
    with OUT.open("w", encoding="utf-8") as out:
        for idx, r in enumerate(rows, start=1):
            if norm(r.get("status")).lower() != "ok" or idx not in LABELS:
                rec = {
                    "source_file": SRC.name, "label_file": OUT.name,
                    "input_url": r.get("input_url"), "url": r.get("url"), "url_hash": r.get("url_hash"),
                    "domain": r.get("domain"), "title": r.get("title"),
                    "status": "error", "source_status": norm(r.get("status")),
                    "teacher_model": MODEL, "error_code": "source_status_error",
                    "message": "Source content record status is not ok; categorization skipped.",
                    "teacher_model_details": md,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); err += 1; continue

            enriched = []
            for rank_id, score in LABELS[idx]:
                trow = by_id.get(rank_id)
                if trow is None:
                    missing_ids.add(rank_id); continue
                enriched.append({
                    "unique_id": trow["unique_id"], "parent_id": trow["parent_id"],
                    "path": trow["path"], "score": round(float(score), 6),
                    "llm_rank": len(enriched) + 1,
                })
            enriched = enriched[:5]
            ids = [c["unique_id"] for c in enriched]
            primary = enriched[0] if enriched else {}
            rec = {
                "source_file": SRC.name, "label_file": OUT.name,
                "input_url": r.get("input_url"), "url": r.get("url"), "url_hash": r.get("url_hash"),
                "domain": r.get("domain"), "title": r.get("title"),
                "meta_description": r.get("meta_description"), "headings": r.get("headings") or [],
                "body_text": norm(r.get("body_text"))[:MAX_BODY], "page_text": page_text(r),
                "teacher_model": MODEL, "teacher_top_k": len(enriched),
                "teacher_top_categories": enriched, "teacher_top_category_ids": ids,
                "teacher_primary_category_id": norm(primary.get("unique_id")),
                "teacher_primary_category_path": norm(primary.get("path")),
                "teacher_primary_score": primary.get("score"),
                "teacher_score_gap_top1_top2": score_gap(enriched),
                "confidence_bucket": bucket(enriched),
                "teacher_usage": {}, "teacher_model_details": md,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); ok += 1
    if missing_ids:
        raise SystemExit(f"ERROR: unknown taxonomy ids referenced: {sorted(missing_ids)}")
    print(f"Wrote {OUT}  ok={ok} error={err} total={ok+err}")


if __name__ == "__main__":
    main()
