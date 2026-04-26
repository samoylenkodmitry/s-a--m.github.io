#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMP_ABORT_CODE = 75


def log(message: str) -> None:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def cpu_temp_c() -> float | None:
    result = subprocess.run(
        ["sensors", "-j"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    raw = result.stdout
    start = raw.find("{")
    if start < 0:
        return None
    try:
        data = json.loads(raw[start:])
    except json.JSONDecodeError:
        return None

    temps: list[float] = []
    for chip_name, chip in data.items():
        if not isinstance(chip, dict):
            continue
        if not (chip_name.startswith("k10temp") or chip_name.startswith("acpitz")):
            continue
        for group in chip.values():
            if not isinstance(group, dict):
                continue
            for key, value in group.items():
                if key.endswith("_input") and isinstance(value, (int, float)):
                    temps.append(float(value))
    return max(temps) if temps else None


def pending_exists(args: argparse.Namespace) -> bool:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_leetcode_ai.py",
            "--dry-run",
            "--limit",
            "1",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        log(result.stdout.strip())
        return True
    return "No LeetCode AI analysis is pending." not in result.stdout


def run_batch(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-u",
        "scripts/analyze_leetcode_ai.py",
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--limit",
        str(args.batch_size),
        "--timeout",
        str(args.timeout),
        "--request-delay",
        str(args.request_delay),
    ]
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    if args.response_format:
        command.extend(["--response-format", args.response_format])
    if args.no_think:
        command.append("--no-think")
    if args.fallback_on_error:
        command.append("--fallback-on-error")
    if args.max_tokens:
        command.extend(["--max-tokens", str(args.max_tokens)])

    process = subprocess.Popen(command, cwd=ROOT)
    while True:
        returncode = process.poll()
        if returncode is not None:
            return returncode

        time.sleep(args.temp_poll)
        temp = cpu_temp_c()
        if temp is None or temp < args.max_temp:
            continue

        log(f"CPU temp {temp:.1f}C >= {args.max_temp:g}C while analyzer is running; stopping batch")
        process.terminate()
        try:
            process.wait(timeout=args.kill_grace)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return TEMP_ABORT_CODE


def sleep_with_reason(seconds: float, reason: str) -> None:
    log(f"{reason}; sleeping {seconds:g}s")
    time.sleep(seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LeetCode AI analysis with temperature cooldown pauses.")
    parser.add_argument("--provider", default="lmstudio")
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="google/gemma-4-e2b")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--request-delay", type=float, default=0)
    parser.add_argument("--response-format", choices=("json_schema", "json_object", "none"), default="json_schema")
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--fallback-on-error", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--max-temp", type=float, default=94.0)
    parser.add_argument("--warm-temp", type=float, default=88.0)
    parser.add_argument("--cooldown", type=float, default=300.0)
    parser.add_argument("--warm-delay", type=float, default=120.0)
    parser.add_argument("--between", type=float, default=45.0)
    parser.add_argument("--error-delay", type=float, default=120.0)
    parser.add_argument("--temp-poll", type=float, default=5.0)
    parser.add_argument("--kill-grace", type=float, default=10.0)
    args = parser.parse_args()

    log(
        "Starting guarded AI backfill "
        f"model={args.model} batch={args.batch_size} max_temp={args.max_temp:g}C"
    )
    while pending_exists(args):
        temp = cpu_temp_c()
        if temp is not None and temp >= args.max_temp:
            sleep_with_reason(args.cooldown, f"CPU temp {temp:.1f}C >= {args.max_temp:g}C")
            continue

        if temp is None:
            log("CPU temp unavailable; running one batch")
        else:
            log(f"CPU temp {temp:.1f}C; running one batch")

        returncode = run_batch(args)
        if returncode == TEMP_ABORT_CODE:
            sleep_with_reason(args.cooldown, "Batch stopped for temperature guard")
            continue
        if returncode != 0:
            sleep_with_reason(args.error_delay, f"Analyzer exited {returncode}")
            continue

        temp = cpu_temp_c()
        if temp is not None and temp >= args.max_temp:
            sleep_with_reason(args.cooldown, f"CPU temp {temp:.1f}C >= {args.max_temp:g}C after batch")
        elif temp is not None and temp >= args.warm_temp:
            sleep_with_reason(args.warm_delay, f"CPU temp {temp:.1f}C >= {args.warm_temp:g}C after batch")
        else:
            sleep_with_reason(args.between, "Batch complete")

    log("No LeetCode AI analysis is pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
