#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "_leetcode_source" / "2023-07-14-leetcode_daily.md"
DATA_PATH = ROOT / "_data" / "leetcode_library.json"
PATTERN_DIR = ROOT / "leetcode" / "pattern"
YEAR_DIR = ROOT / "leetcode" / "year"
PROBLEM_DIR = ROOT / "leetcode" / "problem"
RAW_ARCHIVE_URL = "/2023/07/14/leetcode_daily.html"

TITLE_LINK_RE = re.compile(r"^\[([^\]]+)\]\(([^)]*)\)\s*(.*)$")
DATE_RE = re.compile(r"^# (\d{1,2})\.(\d{1,2})\.(\d{4})$")
NUMERIC_TITLE_RE = re.compile(r"^(\d+)\.\s*(.+)$")
URL_TITLE_RE = re.compile(r"^https?://leetcode\.com/problems/([^/]+)/?")
TELEGRAM_RE = re.compile(r"https://t\.me/leetcode_daily_unstoppable/\d+")
IMAGE_RE = re.compile(r"/assets/leetcode_daily_images/[^)\s]+")
HASHTAG_RE = re.compile(r"(?<!\w)#([a-z0-9-]+)", re.IGNORECASE)

AUX_LINK_LABELS = {"substack", "youtube", "blog post"}


@dataclass(frozen=True)
class PatternRule:
    slug: str
    label: str
    regexes: tuple[str, ...]


PATTERN_RULES = (
    PatternRule("union-find", "Union Find", (r"\bunion[ -]?find\b", r"\buf\b")),
    PatternRule("dynamic-programming", "Dynamic Programming", (r"\bdp\b", r"\bmemo(?:ization)?\b", r"\bknapsack\b")),
    PatternRule("graph", "Graph", (r"\bgraph\b", r"\bbfs\b", r"\bdfs\b", r"\bdijkstra\b", r"\btopological\b")),
    PatternRule("tree", "Tree", (r"\bbinary tree\b", r"\bbst\b", r"\btree\b")),
    PatternRule("trie", "Trie", (r"\btrie\b",)),
    PatternRule("backtracking", "Backtracking", (r"\bbacktracking\b", r"\bbacktrack\b")),
    PatternRule("binary-search", "Binary Search", (r"\bbinary search\b",)),
    PatternRule("prefix-sum", "Prefix Sum", (r"\bprefix[ -]?sum\b",)),
    PatternRule("two-pointers", "Two Pointers", (r"\btwo pointers?\b",)),
    PatternRule("sliding-window", "Sliding Window", (r"\bsliding window\b",)),
    PatternRule("monotonic-stack", "Monotonic Stack", (r"\bmonotonic stack\b",)),
    PatternRule("stack", "Stack", (r"\bstack\b",)),
    PatternRule("queue", "Queue", (r"\bqueue\b", r"\bdeque\b")),
    PatternRule("heap", "Heap", (r"\bheap\b", r"\bpriority queue\b")),
    PatternRule("greedy", "Greedy", (r"\bgreedy\b",)),
    PatternRule("hash-map", "Hash Map", (r"\bhash map\b", r"\bhashmap\b", r"\bdictionary\b", r"\bcounter\b", r"\bcounts\(\)\b")),
    PatternRule("sorting", "Sorting", (r"\bsort(?:ing)?\b",)),
    PatternRule("linked-list", "Linked List", (r"\blinked list\b",)),
    PatternRule("bit-manipulation", "Bit Manipulation", (r"\bbit(?:s)?\b", r"\bxor\b")),
    PatternRule("simulation", "Simulation", (r"\bsimulation\b", r"\bsimulate\b", r"\bbrute force\b", r"\brobot\b")),
    PatternRule("math", "Math", (r"\bgeometry\b", r"\bmath\b", r"\bprime\b", r"\bgcd\b", r"\blcm\b", r"\bmod(?:ulo)?\b")),
    PatternRule("string", "String", (r"\bstring\b", r"\bpalindrome\b", r"\bword\b", r"\bsubstring\b")),
    PatternRule("array", "Array", (r"\barray\b", r"\bmatrix\b", r"\bgrid\b")),
    PatternRule("intervals", "Intervals", (r"\binterval\b",)),
)

HASHTAG_ALIASES = {
    "uf": ("union-find", "Union Find"),
    "unionfind": ("union-find", "Union Find"),
    "union-find": ("union-find", "Union Find"),
    "dp": ("dynamic-programming", "Dynamic Programming"),
    "memo": ("dynamic-programming", "Dynamic Programming"),
    "graph": ("graph", "Graph"),
    "bfs": ("graph", "Graph"),
    "dfs": ("graph", "Graph"),
    "tree": ("tree", "Tree"),
    "bst": ("tree", "Tree"),
    "trie": ("trie", "Trie"),
    "backtracking": ("backtracking", "Backtracking"),
    "backtrack": ("backtracking", "Backtracking"),
    "binarysearch": ("binary-search", "Binary Search"),
    "binary-search": ("binary-search", "Binary Search"),
    "prefix": ("prefix-sum", "Prefix Sum"),
    "prefixsum": ("prefix-sum", "Prefix Sum"),
    "prefix-sum": ("prefix-sum", "Prefix Sum"),
    "twopointers": ("two-pointers", "Two Pointers"),
    "two-pointers": ("two-pointers", "Two Pointers"),
    "stack": ("stack", "Stack"),
    "queue": ("queue", "Queue"),
    "heap": ("heap", "Heap"),
    "greedy": ("greedy", "Greedy"),
    "hashmap": ("hash-map", "Hash Map"),
    "hash-map": ("hash-map", "Hash Map"),
    "linkedlist": ("linked-list", "Linked List"),
    "linked-list": ("linked-list", "Linked List"),
    "bits": ("bit-manipulation", "Bit Manipulation"),
    "bit": ("bit-manipulation", "Bit Manipulation"),
    "xor": ("bit-manipulation", "Bit Manipulation"),
    "math": ("math", "Math"),
    "simulation": ("simulation", "Simulation"),
    "array": ("array", "Array"),
    "string": ("string", "String"),
    "interval": ("intervals", "Intervals"),
}

DIFFICULTY_ALIASES = {
    "easy": "easy",
    "medium": "medium",
    "meidum": "medium",
    "middle": "medium",
    "hard": "hard",
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "entry"


def title_case_from_slug(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("-"))


def yaml_quote(value: str) -> str:
    return json.dumps(value)


def write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def clean_generated_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file() and child.suffix == ".md":
            child.unlink()


def parse_source_blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    current_date: str | None = None
    current_lines: list[str] = []
    for line in lines:
        match = DATE_RE.match(line)
        if match:
            if current_date is not None:
                blocks.append((current_date, current_lines))
            current_date = ".".join(match.groups())
            current_lines = []
            continue
        if current_date is not None:
            current_lines.append(line)
    if current_date is not None:
        blocks.append((current_date, current_lines))
    return blocks


def first_non_aux_link(lines: Iterable[str]) -> tuple[str, str, str]:
    for line in lines:
        match = TITLE_LINK_RE.match(line.strip())
        if not match:
            continue
        label, url, tail = match.groups()
        if label.lower() in AUX_LINK_LABELS:
            continue
        return label.strip(), url.strip(), tail.strip()
    raise ValueError("missing primary link")


def extract_aux_links(lines: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        match = TITLE_LINK_RE.match(line.strip())
        if not match:
            continue
        label, url, _ = match.groups()
        lowered = label.lower()
        if lowered in AUX_LINK_LABELS:
            result[lowered.replace(" ", "_")] = url.strip()
    return result


def extract_section(lines: list[str], heading: str) -> list[str]:
    marker = f"#### {heading}"
    start = None
    for index, line in enumerate(lines):
        if line.strip() == marker:
            start = index + 1
            break
    if start is None:
        return []
    result: list[str] = []
    fence_open = False
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("#### "):
            break
        if stripped.startswith("```"):
            fence_open = not fence_open
            continue
        if fence_open:
            continue
        result.append(line)
    return result


def strip_markdown(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("*", " ").replace("_", " ")
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_line_for_summary(line: str) -> str:
    text = re.sub(r"^[*\-\d. )]+", "", line.strip())
    text = strip_markdown(text)
    text = HASHTAG_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" -:")
    return text


def extract_takeaway(lines: list[str]) -> str | None:
    candidates = [
        *extract_section(lines, "Problem TLDR"),
        *extract_prefixed_section(lines, "Explanation:"),
        *extract_section(lines, "Approach"),
    ]
    if not candidates:
        candidates = lines[:24]
    for line in candidates:
        stripped = line.strip()
        if not stripped or TITLE_LINK_RE.match(stripped) or stripped.startswith("http") or stripped.startswith("```"):
            continue
        lowered = stripped.lower()
        if lowered.startswith(("solution", "speed:", "memory:", "complexity")):
            continue
        text = clean_line_for_summary(line)
        if text:
            return text
    return None


def extract_prefixed_section(lines: list[str], prefix: str) -> list[str]:
    for index, line in enumerate(lines):
        if not line.strip().startswith(prefix):
            continue
        result = [line.split(prefix, 1)[1].strip()]
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            if not stripped and any(result):
                break
            if not stripped:
                continue
            if stripped.startswith("```"):
                continue
            if stripped.startswith("#### "):
                break
            if stripped.startswith(("Speed:", "Memory:", "Complexity", "# ")):
                break
            result.append(candidate)
        return result
    return []


def infer_title(label: str, url: str) -> tuple[int | None, str]:
    numeric_match = NUMERIC_TITLE_RE.match(label)
    if numeric_match:
        return int(numeric_match.group(1)), numeric_match.group(2).strip()
    url_label_match = URL_TITLE_RE.match(label)
    if url_label_match:
        return None, title_case_from_slug(url_label_match.group(1))
    url_match = URL_TITLE_RE.match(url)
    if url_match and label.startswith("http"):
        return None, title_case_from_slug(url_match.group(1))
    return None, label.strip()


def infer_difficulty(raw: str, lines: list[str]) -> str:
    tokens = [token.strip(" #").lower() for token in raw.split()]
    for token in tokens:
        if token in DIFFICULTY_ALIASES:
            return DIFFICULTY_ALIASES[token]
    combined = " ".join(lines[:12]).lower()
    for token, normalized in DIFFICULTY_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", combined):
            return normalized
    return "medium"


def collect_hashtags(lines: Iterable[str]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in lines:
        for raw_tag in HASHTAG_RE.findall(line.lower()):
            tag = raw_tag.replace("_", "-")
            alias = HASHTAG_ALIASES.get(tag)
            if not alias or alias[0] in seen:
                continue
            found.append(alias)
            seen.add(alias[0])
    return found


def infer_patterns(title: str, takeaway: str | None, lines: list[str]) -> tuple[list[str], list[str]]:
    slug_to_label: dict[str, str] = {}
    signal_lines = [title]
    if takeaway:
        signal_lines.append(takeaway)
    signal_lines.extend(extract_section(lines, "Problem TLDR"))
    signal_lines.extend(extract_section(lines, "Intuition")[:12])
    signal_lines.extend(lines[:18])

    for slug, label in collect_hashtags(signal_lines):
        slug_to_label.setdefault(slug, label)
        if len(slug_to_label) == 3:
            break

    normalized = strip_markdown(" ".join(signal_lines)).lower()
    for rule in PATTERN_RULES:
        if rule.slug in slug_to_label:
            continue
        if any(re.search(regex, normalized) for regex in rule.regexes):
            slug_to_label[rule.slug] = rule.label
        if len(slug_to_label) == 3:
            break

    slugs = list(slug_to_label)
    if not slugs:
        slugs = ["implementation"]
        slug_to_label["implementation"] = "Implementation"
    labels = [slug_to_label[slug] for slug in slugs]
    return slugs, labels


def infer_languages(lines: list[str], problem_url: str, blog_post_url: str | None) -> list[str]:
    combined = "\n".join(lines).lower()
    languages: list[str] = []
    if "```kotlin" in combined or "kotlin" in (blog_post_url or "").lower() or "kotlin" in problem_url.lower():
        languages.append("kotlin")
    if "```rust" in combined or "rust" in (blog_post_url or "").lower() or "rust" in problem_url.lower():
        languages.append("rust")
    return languages


def build_entry(date_display: str, lines: list[str]) -> dict[str, object]:
    day, month, year = (int(part) for part in date_display.split("."))
    date_iso = f"{year:04d}-{month:02d}-{day:02d}"
    label, problem_url, difficulty_tail = first_non_aux_link(lines)
    aux_links = extract_aux_links(lines)
    problem_id, title = infer_title(label, problem_url)
    difficulty = infer_difficulty(difficulty_tail, lines)
    takeaway = extract_takeaway(lines)
    pattern_slugs, pattern_labels = infer_patterns(title, takeaway, lines)
    blog_post_url = aux_links.get("blog_post")
    languages = infer_languages(lines, problem_url, blog_post_url)
    solution_url = blog_post_url or (problem_url if "/solutions/" in problem_url else None)
    image_match = IMAGE_RE.search("\n".join(lines))
    telegram_match = TELEGRAM_RE.search("\n".join(lines))
    slug_base = f"{problem_id}-{title}" if problem_id is not None else title
    slug = f"{date_iso}-{slugify(slug_base)}"
    display_title = f"{problem_id}. {title}" if problem_id is not None else title

    kotlin_url = solution_url if "kotlin" in languages else None
    rust_url = solution_url if "rust" in languages else None

    return {
        "slug": slug,
        "date": date_iso,
        "display_date": date_display,
        "year": f"{year:04d}",
        "problem_id": problem_id,
        "title": title,
        "display_title": display_title,
        "difficulty": difficulty,
        "pattern_slugs": pattern_slugs,
        "pattern_labels": pattern_labels,
        "languages": languages,
        "problem_url": problem_url,
        "kotlin_url": kotlin_url,
        "rust_url": rust_url,
        "blog_post_url": blog_post_url,
        "substack_url": aux_links.get("substack"),
        "youtube_url": aux_links.get("youtube"),
        "telegram_url": telegram_match.group(0) if telegram_match else None,
        "image_url": image_match.group(0) if image_match else None,
        "takeaway": takeaway,
        "page_url": f"/leetcode/problem/{slug}/",
        "raw_archive_url": RAW_ARCHIVE_URL,
    }


def build_library(entries: list[dict[str, object]]) -> dict[str, object]:
    pattern_counts: Counter[str] = Counter()
    pattern_labels: dict[str, str] = {}
    year_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    rust_count = 0

    for entry in entries:
        year_counts[entry["year"]] += 1
        for language in entry["languages"]:
            language_counts[language] += 1
        if "rust" in entry["languages"]:
            rust_count += 1
        for slug, label in zip(entry["pattern_slugs"], entry["pattern_labels"]):
            pattern_counts[slug] += 1
            pattern_labels[slug] = label

    years = [
        {"value": year, "label": year, "count": count, "url": f"/leetcode/year/{year}/"}
        for year, count in sorted(year_counts.items(), reverse=True)
    ]
    languages = [
        {"value": language, "label": language.capitalize(), "count": language_counts[language], "url": f"/leetcode/language/{language}/"}
        for language in ("kotlin", "rust")
        if language_counts[language]
    ]
    patterns = [
        {"slug": slug, "label": pattern_labels[slug], "count": count, "url": f"/leetcode/pattern/{slug}/"}
        for slug, count in sorted(
            pattern_counts.items(),
            key=lambda item: (item[0] == "implementation", -item[1], pattern_labels[item[0]]),
        )
    ]

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_url": RAW_ARCHIVE_URL,
        "stats": {
            "entry_count": len(entries),
            "year_count": len(years),
            "pattern_count": len(patterns),
            "rust_count": rust_count,
        },
        "years": years,
        "languages": languages,
        "patterns": patterns,
        "entries": entries,
    }


def render_year_page(year: dict[str, object]) -> str:
    archive_title = f"{year['label']} archive"
    summary = f"{year['count']} entries from {year['label']}."
    return "\n".join(
        [
            "---",
            "layout: leetcode-list",
            f"title: {yaml_quote(archive_title)}",
            f"permalink: {yaml_quote(year['url'])}",
            "leetcode_ui: true",
            "library_kind: year",
            "library_label: Year",
            f"library_heading: {yaml_quote(archive_title)}",
            f"library_intro: {yaml_quote('A year-sized slice of the daily archive, ordered newest first.')}",
            f"library_summary: {yaml_quote(summary)}",
            f"year_value: {yaml_quote(year['value'])}",
            "---",
            "",
        ]
    )


def render_pattern_page(pattern: dict[str, object]) -> str:
    page_title = f"{pattern['label']} pattern"
    summary = f"{pattern['count']} entries tagged as {pattern['label']}."
    return "\n".join(
        [
            "---",
            "layout: leetcode-list",
            f"title: {yaml_quote(page_title)}",
            f"permalink: {yaml_quote(pattern['url'])}",
            "leetcode_ui: true",
            "library_kind: pattern",
            "library_label: Pattern",
            f"library_heading: {yaml_quote(pattern['label'])}",
            f"library_intro: {yaml_quote('A compact view of archive entries grouped by recurring technique.')}",
            f"library_summary: {yaml_quote(summary)}",
            f"pattern_slug: {yaml_quote(pattern['slug'])}",
            "---",
            "",
        ]
    )


def render_problem_page(entry: dict[str, object]) -> str:
    return "\n".join(
        [
            "---",
            "layout: leetcode-entry",
            f"title: {yaml_quote(entry['display_title'])}",
            f"permalink: {yaml_quote(entry['page_url'])}",
            "leetcode_ui: true",
            f"entry_slug: {yaml_quote(entry['slug'])}",
            "---",
            "",
        ]
    )


def main() -> None:
    lines = SOURCE_PATH.read_text(encoding="utf-8").splitlines()
    entries = [build_entry(date_display, block) for date_display, block in parse_source_blocks(lines)]
    library = build_library(entries)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(library, indent=2, sort_keys=False), encoding="utf-8")

    clean_generated_dir(PATTERN_DIR)
    clean_generated_dir(YEAR_DIR)
    clean_generated_dir(PROBLEM_DIR)

    for year in library["years"]:
        write_text(YEAR_DIR / f"{year['value']}.md", render_year_page(year))
    for pattern in library["patterns"]:
        write_text(PATTERN_DIR / f"{pattern['slug']}.md", render_pattern_page(pattern))
    for entry in library["entries"]:
        write_text(PROBLEM_DIR / f"{entry['slug']}.md", render_problem_page(entry))

    print(
        f"Generated {len(entries)} entries, "
        f"{len(library['patterns'])} pattern pages, "
        f"{len(library['years'])} year pages."
    )


if __name__ == "__main__":
    main()
