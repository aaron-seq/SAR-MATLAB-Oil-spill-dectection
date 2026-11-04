"""Check real loopback HTTP batch jobs; optionally measure a bounded workload.

Default: one valid PNG, terminal result validation and 100 fresh status polls.
For matched worker comparisons use --jobs 6 --concurrency 2 --rounds 3 --polls 1;
only the known-broken comparison server should use --allow-missing-status.
Timings include upload, detection and polling, and do not establish capacity.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def png() -> bytes:
    """A deterministic 64x64 greyscale sea with a dark central patch."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = struct.pack(">I", zlib.crc32(kind + data))
        return struct.pack(">I", len(data)) + kind + data + checksum

    pixels = b"".join(
        b"\0" + bytes(35 if 20 < x < 44 and 20 < y < 44 else 140 + (x * 7 + y * 3) % 80
                      for x in range(64)) for y in range(64)
    )
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", 64, 64, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def request(url: str, data: bytes | None = None) -> tuple[int, dict[str, Any]]:
    headers = {"Connection": "close"}  # New TCP connection for every status poll.
    if data is not None:
        headers["Content-Type"] = "multipart/form-data; boundary=batch-runtime-check"
    req = Request(url, data=data, headers=headers)
    try:
        with urlopen(req, timeout=10) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def wait_ready(url: str) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            status, body = request(url + "/health")
            if status == 200 and body.get("detector_ready") is True:
                return
        except (URLError, TimeoutError):
            pass  # Startup is the only place connection failures are retried.
        time.sleep(0.2)
    raise RuntimeError("server did not become ready within 180 seconds")


def job(url: str, payload: bytes, polls: int, allow_missing: bool) -> tuple[float, int]:
    started = time.monotonic()
    status, accepted = request(url + "/api/v1/batch-process", payload)
    if status != 200 or not accepted.get("batch_id"):
        raise RuntimeError(f"batch was not accepted: HTTP {status}, {accepted}")
    batch_id = accepted["batch_id"]
    deadline = started + 60
    missing = 0
    completed_polls = 0
    latency = 0.0
    while time.monotonic() < deadline:
        status, body = request(url + "/api/v1/batch-status/" + batch_id)
        if status == 404 and allow_missing:
            missing += 1
        elif status != 200 or body.get("batch_id") != batch_id:
            raise RuntimeError(f"retained job status failed: HTTP {status}, {body}")
        elif body.get("status") == "completed":
            results = body.get("results", [])
            if (body.get("total_images") != 1 or body.get("completed") != 1
                    or body.get("error") is not None or len(results) != 1
                    or results[0].get("success") is not True or results[0].get("error")
                    or results[0].get("filename") != "scene.png"
                    or not results[0].get("results")):
                raise RuntimeError(f"invalid completed batch: {body}")
            summary = results[0]["results"]
            if (not isinstance(summary, dict) or summary.get("method") != "adaptive_threshold"
                    or not isinstance(summary.get("oil_spill_detected"), bool)
                    or type(summary.get("affected_area_pixels")) is not int
                    or not 0 <= summary["affected_area_pixels"] <= 512 * 512):
                raise RuntimeError(f"invalid detection summary: {summary}")
            for name, upper in (("coverage_percent", 100), ("confidence_score", 1),
                                ("processing_time_seconds", math.inf)):
                value = summary.get(name)
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value)
                        or not 0 <= value <= upper):
                    raise RuntimeError(f"invalid detection field {name}: {value}")
            if completed_polls == 0:
                latency = time.monotonic() - started
            completed_polls += 1
            if completed_polls >= polls:
                return latency, missing
        elif body.get("status") != "processing":
            raise RuntimeError(f"batch entered unexpected state: {body}")
        else:
            time.sleep(0.01)
    raise RuntimeError(f"batch did not complete with {polls} successful polls within 60 seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--polls", type=int, default=100)
    parser.add_argument("--allow-missing-status", action="store_true")
    args = parser.parse_args()
    parsed = urlsplit(args.url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        parser.error("--url must be an HTTP loopback origin without credentials")
    for name, limit in (("jobs", 10), ("concurrency", 4), ("rounds", 5), ("polls", 100)):
        if not 1 <= getattr(args, name) <= limit:
            parser.error(f"--{name} must be between 1 and {limit}")
    url = args.url.rstrip("/")
    payload = (b'--batch-runtime-check\r\nContent-Disposition: form-data; name="images"; '
               b'filename="scene.png"\r\nContent-Type: image/png\r\n\r\n' + png()
               + b"\r\n--batch-runtime-check--\r\n")
    wait_ready(url)
    status, _ = request(url + "/api/v1/batch-status/" + str(uuid.uuid4()))
    if status != 404:
        raise RuntimeError(f"unknown UUID must return 404, got {status}")
    job(url, payload, 1, args.allow_missing_status)  # Identical unmeasured warmup.
    rates: list[float] = []
    latencies: list[float] = []
    for round_number in range(args.rounds):
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            outcomes = list(pool.map(
                lambda _: job(url, payload, args.polls, args.allow_missing_status), range(args.jobs)
            ))
        elapsed = time.monotonic() - started
        rates.append(len(outcomes) / elapsed)
        latencies.extend(outcome[0] for outcome in outcomes)
        print(json.dumps({"round": round_number + 1, "completed_count": len(outcomes),
                          "elapsed_s": elapsed, "jobs_per_s": rates[-1],
                          "latencies_s": [outcome[0] for outcome in outcomes],
                          "status_404_count": sum(outcome[1] for outcome in outcomes)}), flush=True)
    print(json.dumps({"summary": True, "completed_count": len(latencies),
                      "concurrency": args.concurrency, "warmup_jobs": 1,
                      "polls_per_completed_job": args.polls,
                      "allow_missing_status": args.allow_missing_status,
                      "mean_jobs_per_s": statistics.mean(rates),
                      "variance_jobs_per_s": statistics.pvariance(rates),
                      "mean_latency_s": statistics.mean(latencies),
                      "variance_latency_s": statistics.pvariance(latencies)}), flush=True)


if __name__ == "__main__":
    main()
