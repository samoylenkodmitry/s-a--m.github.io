#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_leetcode_library import (
    AI_ANALYSIS_PATH,
    AI_PROMPT_VERSION,
    SOURCE_PATH,
    build_entry,
    date_display_to_iso,
    parse_source_blocks,
    slugify,
    source_block_hash,
    source_block_text,
)


TAG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "slug": {"type": "string"},
        "label": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["slug", "label", "confidence"],
}

AI_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "data_structures": {"type": "array", "items": TAG_SCHEMA, "maxItems": 8},
        "algorithms": {"type": "array", "items": TAG_SCHEMA, "maxItems": 8},
        "techniques": {"type": "array", "items": TAG_SCHEMA, "maxItems": 8},
        "domains": {"type": "array", "items": TAG_SCHEMA, "maxItems": 6},
        "summary": {"type": "string", "maxLength": 220},
        "manual_review": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "data_structures",
        "algorithms",
        "techniques",
        "domains",
        "summary",
        "manual_review",
        "confidence",
    ],
}


@dataclass(frozen=True)
class WorkItem:
    reason: str
    date_display: str
    block_lines: list[str]
    entry: dict[str, object]
    source_hash: str


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "prompt_version": AI_PROMPT_VERSION, "blocks": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    data.setdefault("version", 1)
    data.setdefault("prompt_version", AI_PROMPT_VERSION)
    data.setdefault("blocks", {})
    if not isinstance(data["blocks"], dict):
        raise ValueError(f"{path} field 'blocks' must be an object")
    return data


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    blocks = cache.get("blocks", {})
    cache["version"] = 1
    cache["prompt_version"] = AI_PROMPT_VERSION
    cache["updated_at"] = now_utc()
    cache["blocks"] = {key: blocks[key] for key in sorted(blocks, reverse=True)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def normalize_date_arg(value: str) -> str:
    if "-" in value:
        return value
    return date_display_to_iso(value)


def cache_record_for_entry(blocks: dict[str, Any], entry: dict[str, object]) -> dict[str, Any] | None:
    slug_key = str(entry["slug"])
    record = blocks.get(slug_key)
    if isinstance(record, dict):
        return record

    date_record = blocks.get(str(entry["date"]))
    if isinstance(date_record, dict) and date_record.get("slug") == slug_key:
        return date_record
    return None


def select_work_items(
    source_path: Path,
    cache: dict[str, Any],
    *,
    force: bool,
    date_filter: str | None,
    include_prompt_stale: bool,
) -> list[WorkItem]:
    source_blocks = parse_source_blocks(source_path.read_text(encoding="utf-8").splitlines())
    blocks = cache["blocks"]
    selected: list[WorkItem] = []
    normalized_date_filter = normalize_date_arg(date_filter) if date_filter else None

    for date_display, block_lines in source_blocks:
        entry = build_entry(date_display, block_lines)
        date_iso = str(entry["date"])
        if normalized_date_filter and date_iso != normalized_date_filter:
            continue

        digest = source_block_hash(date_display, block_lines)
        record = cache_record_for_entry(blocks, entry)
        reason = ""
        if force:
            reason = "forced"
        elif not isinstance(record, dict):
            reason = "missing"
        elif record.get("source_hash") != digest:
            reason = "source-changed"
        elif include_prompt_stale and record.get("prompt_version") != AI_PROMPT_VERSION:
            reason = "prompt-changed"
        elif record.get("status") != "analyzed":
            reason = "not-analyzed"

        if reason:
            selected.append(WorkItem(reason, date_display, block_lines, entry, digest))

    return selected


def endpoint_for_provider(provider: str, base_url: str | None) -> str:
    if provider == "github":
        return "https://models.github.ai/inference/chat/completions"

    base = (base_url or "http://localhost:1234/v1").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def api_key_for_provider(provider: str, api_key_env: str | None) -> str:
    if api_key_env:
        return os.environ.get(api_key_env, "")
    if provider == "github":
        return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if provider == "lmstudio":
        return os.environ.get("LMSTUDIO_API_KEY", "")
    return os.environ.get("OPENAI_API_KEY", "")


def response_format_payload(kind: str) -> dict[str, object] | None:
    if kind == "none":
        return None
    if kind == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "leetcode_ai_analysis",
            "schema": AI_SCHEMA,
            "strict": True,
        },
    }


def build_messages(item: WorkItem) -> list[dict[str, str]]:
    entry = item.entry
    system = (
        "You classify one LeetCode daily-note block into concise technique metadata. "
        "Use only the provided note and code. Prefer canonical lowercase slugs. "
        "Return JSON only. Do not include markdown."
    )
    user = f"""
Analyze exactly this one date block.

Return these fields:
- data_structures: concrete structures used, for example union-find, hash-map, heap, queue, graph, tree, trie, stack, matrix, linked-list.
- algorithms: algorithm families used, for example dfs, bfs, a-star, recursion, backtracking, binary-search, dynamic-programming, greedy, sorting, two-pointers, sliding-window, prefix-sum, dijkstra, topological-sort.
- techniques: useful implementation or reasoning techniques, for example coordinate-transform, cycle-handling, modulo-invariant, perimeter-flattening.
- domains: broad domain labels when useful, for example graph, geometry, combinatorics, strings, arrays.
- summary: one compact sentence explaining the main technique.
- manual_review: true when the note is too ambiguous.
- confidence: your overall confidence from 0 to 1.

Date: {entry["date"]}
Title: {entry["display_title"]}
Difficulty: {entry["difficulty"]}
LeetCode URL: {entry["problem_url"]}

----- BEGIN DATE BLOCK -----
{source_block_text(item.date_display, item.block_lines)}
----- END DATE BLOCK -----
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_chat_completion(
    *,
    endpoint: str,
    provider: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, object] | None,
    timeout: int,
) -> str:
    if provider == "github" and not api_key:
        raise RuntimeError("GITHUB_TOKEN or GH_TOKEN is required for the GitHub provider")

    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 800,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/vnd.github+json" if provider == "github" else "application/json",
    }
    if provider == "github":
        headers["X-GitHub-Api-Version"] = "2026-03-10"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AI request failed with HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"AI request failed: {error.reason}") from error

    data = json.loads(raw)
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"AI response did not include choices: {raw[:500]}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    raise RuntimeError(f"AI response content was not a string: {raw[:500]}")


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("analysis"), dict):
        data = data["analysis"]
    if not isinstance(data, dict):
        raise ValueError("AI response must be a JSON object")
    return data


def normalize_confidence(value: object) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return 0.0


def normalize_tag_list(value: object, max_items: int) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("tag fields must be arrays")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in value[:max_items]:
        if isinstance(item, str):
            label = item.strip()
            slug = slugify(label)
            confidence = 0.5
        elif isinstance(item, dict):
            label = str(item.get("label") or item.get("name") or item.get("slug") or "").strip()
            slug = str(item.get("slug") or slugify(label)).strip()
            confidence = normalize_confidence(item.get("confidence"))
        else:
            continue
        if not label or not slug or slug in seen:
            continue
        result.append({"slug": slug, "label": label, "confidence": confidence})
        seen.add(slug)
    return result


def validate_analysis(data: dict[str, Any]) -> dict[str, object]:
    return {
        "data_structures": normalize_tag_list(data.get("data_structures", []), 8),
        "algorithms": normalize_tag_list(data.get("algorithms", []), 8),
        "techniques": normalize_tag_list(data.get("techniques", []), 8),
        "domains": normalize_tag_list(data.get("domains", []), 6),
        "summary": str(data.get("summary") or "").strip()[:220],
        "manual_review": bool(data.get("manual_review", False)),
        "confidence": normalize_confidence(data.get("confidence")),
    }


def analyze_item(args: argparse.Namespace, item: WorkItem) -> dict[str, object]:
    content = call_chat_completion(
        endpoint=endpoint_for_provider(args.provider, args.base_url),
        provider=args.provider,
        api_key=api_key_for_provider(args.provider, args.api_key_env),
        model=args.model,
        messages=build_messages(item),
        response_format=response_format_payload(args.response_format),
        timeout=args.timeout,
    )
    return validate_analysis(parse_json_content(content))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze uncached LeetCode date blocks with a JSON-output LLM.")
    parser.add_argument("--provider", choices=("github", "lmstudio", "openai-compatible"), default="lmstudio")
    parser.add_argument("--model", default=os.environ.get("LEETCODE_AI_MODEL", "local-model"))
    parser.add_argument("--base-url", default=os.environ.get("LEETCODE_AI_BASE_URL"))
    parser.add_argument("--api-key-env", default=os.environ.get("LEETCODE_AI_API_KEY_ENV"))
    parser.add_argument("--response-format", choices=("json_schema", "json_object", "none"), default="json_schema")
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--cache", type=Path, default=AI_ANALYSIS_PATH)
    parser.add_argument("--limit", type=int, default=int(os.environ.get("LEETCODE_AI_LIMIT", "1")), help="0 means no limit")
    parser.add_argument("--date", help="Analyze one date, either YYYY-MM-DD or D.M.YYYY")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-prompt-version", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--soft-fail", action="store_true", help="Print AI errors and keep the workflow green")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--request-delay", type=float, default=float(os.environ.get("LEETCODE_AI_REQUEST_DELAY", "0")))
    parser.add_argument("--retry-delay", type=float, default=float(os.environ.get("LEETCODE_AI_RETRY_DELAY", "120")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("LEETCODE_AI_MAX_RETRIES", "3")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache = load_cache(args.cache)
    items = select_work_items(
        args.source,
        cache,
        force=args.force,
        date_filter=args.date,
        include_prompt_stale=not args.ignore_prompt_version,
    )
    if args.limit > 0:
        items = items[: args.limit]

    if args.dry_run:
        if not items:
            print("No LeetCode AI analysis is pending.")
            return 0
        for item in items:
            print(f"{item.entry['date']} {item.entry['display_title']} [{item.reason}]")
        return 0

    if not items:
        print("No LeetCode AI analysis is pending.")
        return 0

    for index, item in enumerate(items, start=1):
        entry = item.entry
        print(f"Analyzing {index}/{len(items)}: {entry['date']} {entry['display_title']} ({item.reason})")
        retry_count = 0
        while True:
            try:
                analysis = analyze_item(args, item)
                break
            except Exception as error:
                retryable = "HTTP 429" in str(error) or "Too Many Requests" in str(error)
                if retryable and retry_count < args.max_retries:
                    retry_count += 1
                    delay = args.retry_delay * retry_count
                    print(f"Rate limited; retrying in {delay:g}s ({retry_count}/{args.max_retries})", file=sys.stderr)
                    time.sleep(delay)
                    continue
                if not args.soft_fail:
                    raise
                print(f"AI analysis stopped: {error}", file=sys.stderr)
                return 0

        cache["blocks"][str(entry["slug"])] = {
            "status": "analyzed",
            "date": entry["date"],
            "display_date": entry["display_date"],
            "slug": entry["slug"],
            "display_title": entry["display_title"],
            "source_hash": item.source_hash,
            "prompt_version": AI_PROMPT_VERSION,
            "provider": args.provider,
            "model": args.model,
            "analyzed_at": now_utc(),
            "analysis": analysis,
        }
        save_cache(args.cache, cache)
        if args.request_delay > 0 and index < len(items):
            time.sleep(args.request_delay)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
