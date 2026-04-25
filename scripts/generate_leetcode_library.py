#!/usr/bin/env python3

from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "_leetcode_source" / "2023-07-14-leetcode_daily.md"
DATA_PATH = ROOT / "_data" / "leetcode_library.json"
AI_ANALYSIS_PATH = ROOT / "_data" / "leetcode_ai_analysis.json"
PATTERN_DIR = ROOT / "leetcode" / "pattern"
YEAR_DIR = ROOT / "leetcode" / "year"
PROBLEM_DIR = ROOT / "leetcode" / "problem"
COLLECTION_DIR = ROOT / "leetcode" / "collection"
EVOLUTION_DIR = ROOT / "leetcode" / "evolution"
RAW_ARCHIVE_URL = "/2023/07/14/leetcode_daily.html"
AI_PROMPT_VERSION = "leetcode-ai-tags-v1"

TITLE_LINK_RE = re.compile(r"^\[([^\]]+)\]\(([^)]*)\)\s*(.*)$")
DATE_RE = re.compile(r"^# (\d{1,2})\.(\d{1,2})\.(\d{4})$")
NUMERIC_TITLE_RE = re.compile(r"^(\d+)\.\s*(.+)$")
URL_TITLE_RE = re.compile(r"^https?://leetcode\.com/problems/([^/]+)/?")
TELEGRAM_RE = re.compile(r"https://t\.me/leetcode_daily_unstoppable/\d+")
IMAGE_RE = re.compile(r"/assets/leetcode_daily_images/[^)\s]+")
HASHTAG_RE = re.compile(r"(?<!\w)#([a-z0-9-]+)", re.IGNORECASE)
UNSOLVED_RE = re.compile(
    r"\b(?:didn'?t|did\s+not|couldn'?t|could\s+not|can'?t|cannot|failed\s+to)\s+solv(?:e|ed|ing)?\b"
    r"|\bnot\s+able\s+to\s+solv(?:e|ed|ing)?\b"
    r"|\bdidn'?t\s+know\s+how\b"
    r"|\bno\s+ideas\b",
    re.IGNORECASE,
)
GIVE_UP_RE = re.compile(r"\b(?:give\s*up|gave\s+up|giveup)\b", re.IGNORECASE)
HINT_RE = re.compile(r"\b(?:hint|hints|look\s+for\s+solution|others?'?\s+solution|editorial|discussion)\b", re.IGNORECASE)
STUCK_RE = re.compile(
    r"\b(?:wrong\s+answer|tle|mle|stuck|dead\s+end|corner\s+case|debugging|don'?t\s+see|can'?t\s+find)\b",
    re.IGNORECASE,
)
CLOCK_MINUTE_RE = re.compile(r"(?<![\w:.-])(\d+):(\d+)\s*(?:minute|minutes|min)?\b", re.IGNORECASE)
MINUTE_RE = re.compile(r"(?<![\w:.-])(\d+)\s*(?:minute|minutes|min)\b", re.IGNORECASE)
DECIMAL_HOUR_RE = re.compile(r"(?<![\w:.-])(\d+(?:\.\d+)?)\s*(?:hr|hrs|hour|hours)\b", re.IGNORECASE)

AUX_LINK_LABELS = {"substack", "youtube", "blog post"}
SOLUTION_LANGUAGES = {"kotlin", "rust"}
DEEP_THINKING_LINE_THRESHOLD = 30


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


def date_display_to_iso(date_display: str) -> str:
    day, month, year = (int(part) for part in date_display.split("."))
    return f"{year:04d}-{month:02d}-{day:02d}"


def source_block_text(date_display: str, lines: list[str]) -> str:
    return "\n".join([f"# {date_display}", *[line.rstrip() for line in lines]]).strip() + "\n"


def source_block_hash(date_display: str, lines: list[str]) -> str:
    return hashlib.sha256(source_block_text(date_display, lines).encode("utf-8")).hexdigest()


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


def normalize_fence_language(value: str) -> str:
    raw = value.strip().split(maxsplit=1)[0].lower() if value.strip() else ""
    raw = raw.strip("{}[]()")
    if raw.startswith("kotlin"):
        return "kotlin"
    if raw.startswith("rust"):
        return "rust"
    if raw == "j":
        return "j"
    return raw


def iter_code_blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    language = ""
    block_lines: list[str] = []
    fence_open = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if fence_open:
                blocks.append((language, block_lines))
                block_lines = []
                language = ""
                fence_open = False
            else:
                language = normalize_fence_language(stripped[3:])
                block_lines = []
                fence_open = True
            continue
        if fence_open:
            block_lines.append(line.rstrip())

    return blocks


def nonblank_line_count(lines: Iterable[str]) -> int:
    return sum(1 for line in lines if line.strip())


def solution_line_bucket(line_count: int | None) -> str:
    if line_count is None:
        return "unknown"
    if line_count <= 2:
        return "one-liner"
    if line_count <= 6:
        return "tiny"
    if line_count <= 12:
        return "compact"
    if line_count <= 25:
        return "standard"
    return "long"


def text_without_solution_code(lines: list[str]) -> str:
    visible_lines: list[str] = []
    language = ""
    fence_open = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if fence_open:
                fence_open = False
                language = ""
            else:
                language = normalize_fence_language(stripped[3:])
                fence_open = True
            continue
        if fence_open and language in SOLUTION_LANGUAGES:
            continue
        visible_lines.append(line)

    return "\n".join(visible_lines)


def extract_minute_markers(text: str) -> list[int]:
    markers: set[int] = set()
    for hours, minutes in CLOCK_MINUTE_RE.findall(text):
        hour_value = int(hours)
        minute_value = int(minutes)
        if hour_value <= 3 and minute_value < 60:
            markers.add(hour_value * 60 + minute_value)
    for minutes in MINUTE_RE.findall(text):
        markers.add(int(minutes))
    for hours in DECIMAL_HOUR_RE.findall(text):
        markers.add(round(float(hours) * 60))
    return sorted(value for value in markers if 0 < value <= 24 * 60)


def extract_entry_metrics(lines: list[str]) -> dict[str, object]:
    blocks = iter_code_blocks(lines)
    solution_blocks: list[dict[str, object]] = []
    language_line_counts: dict[str, int] = {}
    language_char_counts: dict[str, int] = {}
    thought_lines = 0
    thought_chars = 0

    for language, block in blocks:
        nonblank = [line for line in block if line.strip()]
        char_count = len("\n".join(line.strip() for line in nonblank))
        if language == "j":
            thought_lines += len(nonblank)
            thought_chars += char_count
            continue
        if language not in SOLUTION_LANGUAGES or not nonblank:
            continue
        solution_blocks.append(
            {
                "language": language,
                "line_count": len(nonblank),
                "char_count": char_count,
            }
        )
        previous_lines = language_line_counts.get(language)
        previous_chars = language_char_counts.get(language)
        if previous_lines is None or (len(nonblank), char_count) < (previous_lines, previous_chars or 0):
            language_line_counts[language] = len(nonblank)
            language_char_counts[language] = char_count

    shortest_solution = min(
        solution_blocks,
        key=lambda block: (int(block["line_count"]), int(block["char_count"])),
        default=None,
    )
    reflective_text = text_without_solution_code(lines)
    minute_markers = extract_minute_markers(reflective_text)
    max_minute_marker = max(minute_markers) if minute_markers else None
    unsolved = bool(UNSOLVED_RE.search(reflective_text) or GIVE_UP_RE.search(reflective_text))
    hint_or_stuck = bool(HINT_RE.search(reflective_text) or GIVE_UP_RE.search(reflective_text) or STUCK_RE.search(reflective_text))
    almost_gave_up = bool(minute_markers and hint_or_stuck and ((max_minute_marker or 0) >= 25 or GIVE_UP_RE.search(reflective_text)))
    solution_lines = int(shortest_solution["line_count"]) if shortest_solution else None
    solution_chars = int(shortest_solution["char_count"]) if shortest_solution else None
    bucket = solution_line_bucket(solution_lines)

    return {
        "solution_lines": solution_lines,
        "solution_chars": solution_chars,
        "solution_language": shortest_solution["language"] if shortest_solution else None,
        "solution_line_bucket": bucket,
        "language_line_counts": language_line_counts,
        "language_char_counts": language_char_counts,
        "thought_lines": thought_lines,
        "thought_chars": thought_chars,
        "minute_markers": minute_markers,
        "max_minute_marker": max_minute_marker,
        "deep_thinking": thought_lines >= DEEP_THINKING_LINE_THRESHOLD,
        "unsolved": unsolved,
        "almost_gave_up": almost_gave_up,
        "one_liner": bucket == "one-liner",
    }


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


def normalize_problem_page_lines(lines: list[str]) -> list[str]:
    filtered: list[str] = []
    previous_blank = False
    for line in lines:
        if line.strip() == "https://dmitrysamoylenko.com/2023/07/14/leetcode_daily.html":
            continue
        is_blank = not line.strip()
        if is_blank and previous_blank and filtered and not filtered[-1].strip():
            continue
        filtered.append(line.rstrip())
        previous_blank = is_blank
    return filtered


def build_entry(date_display: str, lines: list[str]) -> dict[str, object]:
    day, month, year = (int(part) for part in date_display.split("."))
    date_iso = date_display_to_iso(date_display)
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
    metrics = extract_entry_metrics(lines)
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
        "metrics": metrics,
        "catalog_tags": [],
        "collection_links": [],
        "page_url": f"/leetcode/problem/{slug}/",
        "raw_archive_url": RAW_ARCHIVE_URL,
    }


def problem_group_key(entry: dict[str, object]) -> str:
    problem_id = entry.get("problem_id")
    if problem_id is not None:
        return str(problem_id)
    return slugify(str(entry["title"]))


def add_catalog_tag(entry: dict[str, object], tag: str) -> None:
    tags = entry.setdefault("catalog_tags", [])
    if isinstance(tags, list) and tag not in tags:
        tags.append(tag)


def insight_entry(entry: dict[str, object]) -> dict[str, object]:
    metrics = entry["metrics"]
    return {
        "slug": entry["slug"],
        "date": entry["date"],
        "display_date": entry["display_date"],
        "year": entry["year"],
        "display_title": entry["display_title"],
        "difficulty": entry["difficulty"],
        "page_url": entry["page_url"],
        "problem_url": entry["problem_url"],
        "image_url": entry["image_url"],
        "takeaway": entry["takeaway"],
        "pattern_slugs": entry["pattern_slugs"],
        "pattern_labels": entry["pattern_labels"],
        "languages": entry["languages"],
        "catalog_tags": entry["catalog_tags"],
        "solution_lines": metrics["solution_lines"],
        "solution_chars": metrics["solution_chars"],
        "solution_language": metrics["solution_language"],
        "solution_line_bucket": metrics["solution_line_bucket"],
        "thought_lines": metrics["thought_lines"],
        "thought_chars": metrics["thought_chars"],
        "max_minute_marker": metrics["max_minute_marker"],
        "unsolved": metrics["unsolved"],
        "almost_gave_up": metrics["almost_gave_up"],
    }


def metric_delta(first: object, last: object) -> int | None:
    if first is None or last is None:
        return None
    return int(last) - int(first)


def days_between(start_iso: str, end_iso: str) -> int:
    return (date.fromisoformat(end_iso) - date.fromisoformat(start_iso)).days


def percent_of(value: object, maximum: int) -> int:
    if not value or maximum <= 0:
        return 0
    return max(2, round(int(value) * 100 / maximum))


def repeat_trend(delta: int | None) -> str:
    if delta is None:
        return "unmeasured"
    if delta < 0:
        return "compressed"
    if delta > 0:
        return "expanded"
    return "steady"


def evolution_slug(entry: dict[str, object]) -> str:
    slug_base = f"{entry['problem_id']}-{entry['title']}" if entry.get("problem_id") is not None else str(entry["title"])
    return slugify(slug_base)


def build_repeat_group(key: str, group_entries: list[dict[str, object]]) -> dict[str, object]:
    timeline = sorted(group_entries, key=lambda entry: str(entry["date"]))
    first = timeline[0]
    last = timeline[-1]
    attempts = [insight_entry(entry) for entry in timeline]
    max_solution_lines = max((int(attempt["solution_lines"] or 0) for attempt in attempts), default=0)
    max_thought_lines = max((int(attempt["thought_lines"] or 0) for attempt in attempts), default=0)
    previous: dict[str, object] | None = None
    for index, attempt in enumerate(attempts):
        attempt["attempt_number"] = index + 1
        attempt["solution_line_percent"] = percent_of(attempt["solution_lines"], max_solution_lines)
        attempt["thought_line_percent"] = percent_of(attempt["thought_lines"], max_thought_lines)
        if previous:
            attempt["days_since_previous"] = days_between(str(previous["date"]), str(attempt["date"]))
            attempt["solution_line_delta_from_previous"] = metric_delta(previous["solution_lines"], attempt["solution_lines"])
            attempt["solution_char_delta_from_previous"] = metric_delta(previous["solution_chars"], attempt["solution_chars"])
            attempt["thought_line_delta_from_previous"] = metric_delta(previous["thought_lines"], attempt["thought_lines"])
        else:
            attempt["days_since_previous"] = None
            attempt["solution_line_delta_from_previous"] = None
            attempt["solution_char_delta_from_previous"] = None
            attempt["thought_line_delta_from_previous"] = None
        previous = attempt

    measured_solutions = [attempt for attempt in attempts if attempt["solution_lines"] is not None]
    best_solution = min(
        measured_solutions,
        key=lambda attempt: (int(attempt["solution_lines"]), int(attempt["solution_chars"] or 999999)),
        default=None,
    )
    largest_solution = max(
        measured_solutions,
        key=lambda attempt: (int(attempt["solution_lines"]), int(attempt["solution_chars"] or 0)),
        default=None,
    )
    line_delta = metric_delta(first["metrics"]["solution_lines"], last["metrics"]["solution_lines"])
    char_delta = metric_delta(first["metrics"]["solution_chars"], last["metrics"]["solution_chars"])
    thought_delta = metric_delta(first["metrics"]["thought_lines"], last["metrics"]["thought_lines"])
    return {
        "key": key,
        "problem_id": first["problem_id"],
        "title": first["title"],
        "display_title": first["display_title"],
        "evolution_slug": evolution_slug(first),
        "evolution_url": f"/leetcode/evolution/{evolution_slug(first)}/",
        "count": len(timeline),
        "first_date": first["display_date"],
        "last_date": last["display_date"],
        "first_sort_date": first["date"],
        "last_sort_date": last["date"],
        "span_days": days_between(str(first["date"]), str(last["date"])),
        "latest_page_url": last["page_url"],
        "solution_line_delta": line_delta,
        "solution_char_delta": char_delta,
        "thought_line_delta": thought_delta,
        "solution_line_trend": repeat_trend(line_delta),
        "solution_char_trend": repeat_trend(char_delta),
        "thought_line_trend": repeat_trend(thought_delta),
        "best_solution": best_solution,
        "largest_solution": largest_solution,
        "max_solution_lines": max_solution_lines,
        "max_thought_lines": max_thought_lines,
        "total_thought_lines": sum(int(attempt["thought_lines"] or 0) for attempt in attempts),
        "stuck_count": sum(1 for attempt in attempts if attempt["unsolved"]),
        "one_liner_count": sum(1 for attempt in attempts if attempt["solution_line_bucket"] == "one-liner"),
        "solved_after_stuck": any(
            bool(earlier["unsolved"]) and any(not bool(later["unsolved"]) for later in attempts[index + 1 :])
            for index, earlier in enumerate(attempts[:-1])
        ),
        "entries": attempts,
    }


def build_collections(insights: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "slug": "repeats",
            "kind": "repeat",
            "label": "Repeat Lab",
            "title": "Repeat Lab",
            "url": "/leetcode/collection/repeats/",
            "count": insights["repeat_problem_count"],
            "summary": "Problems solved on multiple dates, grouped into evolution timelines.",
            "metric_label": "problem timelines",
        },
        {
            "slug": "comebacks",
            "kind": "repeat",
            "label": "Comebacks",
            "title": "Could Not First, Solved Later",
            "url": "/leetcode/collection/comebacks/",
            "count": len(insights["comebacks"]),
            "summary": "Repeat problems where an earlier attempt was marked stuck or unsolved and a later attempt was not.",
            "metric_label": "recoveries",
        },
        {
            "slug": "deep-thinking",
            "kind": "entry",
            "label": "Long Thoughts",
            "title": "Long Thinking Traces",
            "url": "/leetcode/collection/deep-thinking/",
            "count": len(insights["deep_thinking"]),
            "summary": "Entries with long j-code thought sections, usually where the interesting part is the search path.",
            "metric_label": "thought-heavy entries",
        },
        {
            "slug": "unsolved",
            "kind": "entry",
            "label": "Could Not Solve",
            "title": "Could Not Solve Cleanly",
            "url": "/leetcode/collection/unsolved/",
            "count": len(insights["unsolved"]),
            "summary": "Entries that explicitly say the solution did not come cleanly without hints, giving up, or outside help.",
            "metric_label": "honest misses",
        },
        {
            "slug": "almost-gave-up",
            "kind": "entry",
            "label": "Pressure Log",
            "title": "Almost Gave Up",
            "url": "/leetcode/collection/almost-gave-up/",
            "count": len(insights["almost_gave_up"]),
            "summary": "Entries with minute markers near the stopping point, hints, wrong answers, or explicit timeboxing.",
            "metric_label": "timeboxed struggles",
        },
        {
            "slug": "one-liners",
            "kind": "entry",
            "label": "Small Code",
            "title": "One-Liner Gallery",
            "url": "/leetcode/collection/one-liners/",
            "count": len(insights["one_liners"]),
            "summary": "Problems whose shortest Kotlin or Rust solution is two nonblank lines or less.",
            "metric_label": "tiny solutions",
        },
    ]


COLLECTION_LINK_RULES = (
    ("repeat", "Repeat Lab", "/leetcode/collection/repeats/", "Evolution"),
    ("comeback", "Comebacks", "/leetcode/collection/comebacks/", "Evolution"),
    ("deep-thinking", "Long Thoughts", "/leetcode/collection/deep-thinking/", None),
    ("unsolved", "Could Not Solve", "/leetcode/collection/unsolved/", None),
    ("almost-gave-up", "Almost Gave Up", "/leetcode/collection/almost-gave-up/", None),
    ("one-liner", "One-Liners", "/leetcode/collection/one-liners/", None),
)


def build_entry_collection_links(entry: dict[str, object], evolution_url: str | None) -> list[dict[str, object]]:
    tags = set(str(tag) for tag in entry.get("catalog_tags", []))
    links: list[dict[str, object]] = []
    for tag, label, url, detail_label in COLLECTION_LINK_RULES:
        if tag not in tags:
            continue
        link = {
            "slug": tag,
            "label": label,
            "url": url,
        }
        if detail_label and evolution_url:
            link["detail_label"] = detail_label
            link["detail_url"] = evolution_url
        links.append(link)
    return links


def load_ai_analysis_cache() -> dict[str, object]:
    if not AI_ANALYSIS_PATH.exists():
        return {}
    try:
        data = json.loads(AI_ANALYSIS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def normalize_ai_tag_items(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    tags: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str):
            label = item.strip()
            slug = slugify(label)
            confidence = None
        elif isinstance(item, dict):
            label = str(item.get("label") or item.get("name") or item.get("slug") or "").strip()
            slug = str(item.get("slug") or slugify(label)).strip()
            confidence = item.get("confidence")
        else:
            continue
        if not label or not slug or slug in seen:
            continue
        tag: dict[str, object] = {"slug": slug, "label": label}
        if isinstance(confidence, (int, float)):
            tag["confidence"] = max(0, min(1, float(confidence)))
        tags.append(tag)
        seen.add(slug)
    return tags


def normalize_ai_analysis(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    data_structures = normalize_ai_tag_items(value.get("data_structures"))
    algorithms = normalize_ai_tag_items(value.get("algorithms"))
    techniques = normalize_ai_tag_items(value.get("techniques"))
    domains = normalize_ai_tag_items(value.get("domains"))
    summary = str(value.get("summary") or "").strip()
    confidence = value.get("confidence")
    if not (data_structures or algorithms or techniques or domains or summary):
        return None
    analysis: dict[str, object] = {
        "data_structures": data_structures,
        "algorithms": algorithms,
        "techniques": techniques,
        "domains": domains,
        "summary": summary,
        "manual_review": bool(value.get("manual_review", False)),
    }
    if isinstance(confidence, (int, float)):
        analysis["confidence"] = max(0, min(1, float(confidence)))
    return analysis


def ai_cache_record_for_entry(blocks: dict[str, object], entry: dict[str, object]) -> dict[str, object] | None:
    slug_key = str(entry["slug"])
    record = blocks.get(slug_key)
    if isinstance(record, dict):
        return record

    date_record = blocks.get(str(entry["date"]))
    if isinstance(date_record, dict) and date_record.get("slug") == slug_key:
        return date_record
    return None


def attach_ai_analysis(entries: list[dict[str, object]], source_blocks: list[tuple[str, list[str]]]) -> None:
    cache = load_ai_analysis_cache()
    blocks = cache.get("blocks", {})
    if not isinstance(blocks, dict):
        return

    for entry, (date_display, block_lines) in zip(entries, source_blocks):
        record = ai_cache_record_for_entry(blocks, entry)
        if record is None:
            continue
        if record.get("status") != "analyzed":
            continue
        if record.get("source_hash") != source_block_hash(date_display, block_lines):
            continue
        analysis = normalize_ai_analysis(record.get("analysis"))
        if analysis is None:
            continue

        ai_tags: list[dict[str, object]] = []
        seen: set[str] = set()
        for group_name in ("data_structures", "algorithms", "techniques", "domains"):
            for tag in analysis.get(group_name, []):
                if not isinstance(tag, dict):
                    continue
                slug = str(tag.get("slug") or "")
                if not slug or slug in seen:
                    continue
                ai_tags.append({"slug": slug, "label": str(tag.get("label") or slug), "group": group_name})
                seen.add(slug)

        entry["ai"] = {
            "provider": record.get("provider"),
            "model": record.get("model"),
            "prompt_version": record.get("prompt_version"),
            "analyzed_at": record.get("analyzed_at"),
            **analysis,
        }
        entry["ai_tags"] = ai_tags
        entry["ai_tag_slugs"] = [tag["slug"] for tag in ai_tags]
        entry["ai_tag_labels"] = [tag["label"] for tag in ai_tags]


def build_insights(entries: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        metrics = entry["metrics"]
        add_catalog_tag(entry, str(metrics["solution_line_bucket"]))
        if metrics["one_liner"]:
            add_catalog_tag(entry, "one-liner")
        if metrics["deep_thinking"]:
            add_catalog_tag(entry, "deep-thinking")
        if metrics["unsolved"]:
            add_catalog_tag(entry, "unsolved")
        if metrics["almost_gave_up"]:
            add_catalog_tag(entry, "almost-gave-up")
        grouped.setdefault(problem_group_key(entry), []).append(entry)

    repeat_groups = []
    comeback_groups = []
    for key, group_entries in grouped.items():
        if len(group_entries) <= 1:
            continue
        for entry in group_entries:
            add_catalog_tag(entry, "repeat")
        repeat_group = build_repeat_group(key, group_entries)
        repeat_groups.append(repeat_group)

        timeline = sorted(group_entries, key=lambda entry: str(entry["date"]))
        has_comeback = any(
            bool(earlier["metrics"]["unsolved"]) and any(not bool(later["metrics"]["unsolved"]) for later in timeline[index + 1 :])
            for index, earlier in enumerate(timeline[:-1])
        )
        if has_comeback:
            for entry in group_entries:
                add_catalog_tag(entry, "comeback")
            comeback_groups.append(repeat_group)

    line_bucket_order = [
        ("one-liner", "One-liners"),
        ("tiny", "Tiny"),
        ("compact", "Compact"),
        ("standard", "Standard"),
        ("long", "Long"),
        ("unknown", "Unknown"),
    ]
    line_bucket_counts = Counter(str(entry["metrics"]["solution_line_bucket"]) for entry in entries)
    line_buckets = [
        {"slug": slug, "label": label, "count": line_bucket_counts[slug]}
        for slug, label in line_bucket_order
        if line_bucket_counts[slug]
    ]

    repeat_groups.sort(key=lambda group: (int(group["count"]), str(group["last_sort_date"])), reverse=True)
    comeback_groups.sort(key=lambda group: str(group["last_sort_date"]), reverse=True)
    evolution_url_by_key = {str(group["key"]): str(group["evolution_url"]) for group in repeat_groups}
    for entry in entries:
        entry["collection_links"] = build_entry_collection_links(entry, evolution_url_by_key.get(problem_group_key(entry)))

    insights = {
        "unique_problem_count": len(grouped),
        "repeat_problem_count": len(repeat_groups),
        "repeated_entry_count": sum(len(group) for group in grouped.values() if len(group) > 1),
        "line_buckets": line_buckets,
        "repeats": repeat_groups,
        "comebacks": comeback_groups,
        "deep_thinking": [
            insight_entry(entry)
            for entry in sorted(entries, key=lambda entry: (int(entry["metrics"]["thought_lines"]), str(entry["date"])), reverse=True)
            if entry["metrics"]["deep_thinking"]
        ],
        "unsolved": [
            insight_entry(entry)
            for entry in sorted(entries, key=lambda entry: str(entry["date"]), reverse=True)
            if entry["metrics"]["unsolved"]
        ],
        "one_liners": [
            insight_entry(entry)
            for entry in sorted(
                entries,
                key=lambda entry: (
                    int(entry["metrics"]["solution_lines"] or 9999),
                    int(entry["metrics"]["solution_chars"] or 999999),
                    str(entry["date"]),
                ),
            )
            if entry["metrics"]["one_liner"]
        ],
        "almost_gave_up": [
            insight_entry(entry)
            for entry in sorted(
                entries,
                key=lambda entry: (int(entry["metrics"]["max_minute_marker"] or 0), int(entry["metrics"]["thought_lines"]), str(entry["date"])),
                reverse=True,
            )
            if entry["metrics"]["almost_gave_up"]
        ],
    }
    insights["collections"] = build_collections(insights)
    return insights


def build_library(entries: list[dict[str, object]]) -> dict[str, object]:
    pattern_counts: Counter[str] = Counter()
    pattern_labels: dict[str, str] = {}
    year_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    rust_count = 0
    insights = build_insights(entries)

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
            "unique_problem_count": insights["unique_problem_count"],
            "year_count": len(years),
            "pattern_count": len(patterns),
            "rust_count": rust_count,
            "repeat_problem_count": insights["repeat_problem_count"],
            "repeated_entry_count": insights["repeated_entry_count"],
            "deep_thinking_count": len(insights["deep_thinking"]),
            "unsolved_count": len(insights["unsolved"]),
            "one_liner_count": len(insights["one_liners"]),
            "almost_gave_up_count": len(insights["almost_gave_up"]),
            "comeback_count": len(insights["comebacks"]),
            "ai_analyzed_count": sum(1 for entry in entries if entry.get("ai")),
        },
        "years": years,
        "languages": languages,
        "patterns": patterns,
        "insights": insights,
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


def render_problem_page(entry: dict[str, object], lines: list[str]) -> str:
    normalized_lines = normalize_problem_page_lines(lines)
    front_matter = [
        "---",
        "layout: leetcode-entry",
        f"title: {yaml_quote(entry['display_title'])}",
        f"permalink: {yaml_quote(entry['page_url'])}",
        "leetcode_ui: true",
        f"entry_slug: {yaml_quote(entry['slug'])}",
        "---",
        "",
    ]
    return "\n".join(front_matter + normalized_lines + [""])


def render_collection_page(collection: dict[str, object]) -> str:
    return "\n".join(
        [
            "---",
            "layout: leetcode-collection",
            f"title: {yaml_quote(str(collection['title']))}",
            f"permalink: {yaml_quote(str(collection['url']))}",
            "leetcode_ui: true",
            f"collection_slug: {yaml_quote(str(collection['slug']))}",
            "---",
            "",
        ]
    )


def render_evolution_page(group: dict[str, object]) -> str:
    page_title = f"{group['display_title']} evolution"
    return "\n".join(
        [
            "---",
            "layout: leetcode-evolution",
            f"title: {yaml_quote(page_title)}",
            f"permalink: {yaml_quote(str(group['evolution_url']))}",
            "leetcode_ui: true",
            f"evolution_slug: {yaml_quote(str(group['evolution_slug']))}",
            "---",
            "",
        ]
    )


def main() -> None:
    lines = SOURCE_PATH.read_text(encoding="utf-8").splitlines()
    source_blocks = parse_source_blocks(lines)
    entries = [build_entry(date_display, block) for date_display, block in source_blocks]
    attach_ai_analysis(entries, source_blocks)
    library = build_library(entries)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(library, indent=2, sort_keys=False), encoding="utf-8")

    clean_generated_dir(PATTERN_DIR)
    clean_generated_dir(YEAR_DIR)
    clean_generated_dir(PROBLEM_DIR)
    clean_generated_dir(COLLECTION_DIR)
    clean_generated_dir(EVOLUTION_DIR)

    for year in library["years"]:
        write_text(YEAR_DIR / f"{year['value']}.md", render_year_page(year))
    for pattern in library["patterns"]:
        write_text(PATTERN_DIR / f"{pattern['slug']}.md", render_pattern_page(pattern))
    for entry, (_, block_lines) in zip(library["entries"], source_blocks):
        write_text(PROBLEM_DIR / f"{entry['slug']}.md", render_problem_page(entry, block_lines))
    for collection in library["insights"]["collections"]:
        write_text(COLLECTION_DIR / f"{collection['slug']}.md", render_collection_page(collection))
    for group in library["insights"]["repeats"]:
        write_text(EVOLUTION_DIR / f"{group['evolution_slug']}.md", render_evolution_page(group))

    print(
        f"Generated {len(entries)} entries, "
        f"{len(library['patterns'])} pattern pages, "
        f"{len(library['years'])} year pages, "
        f"{len(library['insights']['collections'])} collection pages, "
        f"{len(library['insights']['repeats'])} evolution pages."
    )


if __name__ == "__main__":
    main()
