"""Bounded HTTP load probe. Defaults to metadata (no model/API charges).

Set LOAD_TEST_PASSWORD or enter it at the prompt. Never pass a password on the CLI.
Agent QA requires --mode qa explicitly and can incur model charges.
"""
import argparse
import asyncio
import getpass
import json
import math
import os
import time
from collections import Counter

import httpx


def percentile(values, fraction):
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * fraction) - 1)], 2) if ordered else None


async def probe(base_url, username, password, mode, count, concurrency, timeout):
    transport_limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout, limits=transport_limits) as client:
        response = await client.post("/api/auth/login", json={"username": username, "password": password})
        response.raise_for_status()
        headers = {"Authorization": "Bearer " + response.json()["access_token"]}
        semaphore = asyncio.Semaphore(concurrency)
        statuses = Counter()
        latencies = []
        success_latencies = []
        app_failures = 0
        completed = 0
        health = []

        async def request_one():
            nonlocal app_failures, completed
            async with semaphore:
                start = time.perf_counter()
                succeeded = False
                try:
                    if mode == "qa":
                        result = await client.post("/api/rag/agent/qa", headers=headers, json={
                            "question": "知识库涉及哪些内容？", "max_attempts": 1, "max_tool_rounds": 1,
                        })
                    else:
                        result = await client.get("/api/rag/agent/route-preview", headers=headers, params={"question": "知识库涉及哪些内容？"})
                    statuses[str(result.status_code)] += 1
                    succeeded = result.is_success
                    if result.is_success and result.json().get("success") is False:
                        app_failures += 1
                        succeeded = False
                except httpx.HTTPError as exc:
                    statuses[type(exc).__name__] += 1
                finally:
                    elapsed = (time.perf_counter() - start) * 1000
                    latencies.append(elapsed)
                    if succeeded:
                        success_latencies.append(elapsed)
                    completed += 1

        async def health_probe():
            # Independent connection pool so client-side queueing does not masquerade as server lag.
            async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=5) as monitor:
                while completed < count:
                    started = time.perf_counter()
                    try:
                        result = await monitor.get("/api/health")
                        health.append({"status": result.status_code, "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
                    except httpx.HTTPError:
                        health.append({"status": "error"})
                    await asyncio.sleep(0.1)

        started = time.perf_counter()
        try:
            await asyncio.gather(health_probe(), *(request_one() for _ in range(count)))
        finally:
            await client.post("/api/auth/logout", headers=headers)
        elapsed = time.perf_counter() - started
        return {"mode": mode, "requests": count, "concurrency": concurrency,
                "statuses": dict(statuses), "application_failures": app_failures,
                "elapsed_seconds": round(elapsed, 3),
                "completed_rps": round(count / elapsed, 2),
                "successful_rps": round(len(success_latencies) / elapsed, 2),
                "all_latency_ms": {"p50": percentile(latencies, .5), "p95": percentile(latencies, .95)},
                "successful_latency_ms": {"p50": percentile(success_latencies, .5), "p95": percentile(success_latencies, .95)},
                "health_probes": health,
                "note": "Single user; per-user limits apply. Metadata throughput is not LLM throughput."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", required=True)
    parser.add_argument("--mode", choices=["metadata", "qa"], default="metadata")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if not 1 <= args.requests <= 10000 or not 1 <= args.concurrency <= 100:
        parser.error("requests must be 1..10000 and concurrency 1..100")
    password = os.getenv("LOAD_TEST_PASSWORD") or getpass.getpass("Password: ")
    report = asyncio.run(probe(args.base_url, args.username, password, args.mode, args.requests, args.concurrency, args.timeout))
    formatted = json.dumps(report, ensure_ascii=False, indent=2)
    print(formatted)
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(formatted, encoding="utf-8")


if __name__ == "__main__":
    main()
