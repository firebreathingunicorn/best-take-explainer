#!/usr/bin/env python3
"""Best-take explainer — NVIDIA Nemotron on Nebius Token Factory.

Companion service for the Mily group-photo app (Nebius x NVIDIA Global AI
Hackathon, Best Apps and Agents track). Mily's engine already runs fully
on-device: it samples a burst, scores every person's expression per frame,
picks a base frame, and transplants each person's best moment into it —
least-invasive-first, with artifact and identity checks that leave a person
unchanged rather than ship a risky edit.

This tool takes the decision report that engine already writes
(`mily process <burst> --report report.json`) and asks **NVIDIA Nemotron 3
Super**, served on **Nebius Token Factory**, to explain the result like a
human would:

    WHY:     why this photo is everyone's best take (who was lifted from a
             better moment, who was left as shot and why)
    CAPTION: one short share caption for the photo

Privacy: the ONLY data that leaves the device is the anonymous report
numbers — frame indices, check outcomes, scores. No pixels, no faces, no
names. The request is a plain OpenAI-compatible chat completion:

    POST https://api.tokenfactory.nebius.com/v1/chat/completions
    Authorization: Bearer $NEBIUS_API_KEY

Usage:
    python3 explainer.py report.json
    python3 explainer.py report.json --model nvidia/Nemotron-3-Ultra-550b-a55b
    python3 explainer.py example/report.example.json

Model tiers on Token Factory (Oct 2026):
    nvidia/nemotron-3-super-120b-a12b   default — fast everyday calls (256K)
    nvidia/Nemotron-3-Ultra-550b-a55b   heavy reasoner (1M context)

Cost: ~0.9k tokens per explanation. At Nebius's blended $0.36 / 1M tokens
for Super that is ~$0.0003 per photo — about $3 per 10,000 explanations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
ENDPOINT = "https://api.tokenfactory.nebius.com/v1/chat/completions"

SYSTEM_PROMPT = """You are Mily's explainer. Mily fixes group photos: it samples a \
short burst, scores every person's expression per frame (smile, open eyes, \
not mid-word), picks a base frame, and transplants each person's best moment \
into it on-device — least-invasive-first, with artifact and identity checks \
that leave a person unchanged rather than ship a risky edit. You get an \
anonymous decision report: frame indices are 0-based, person labels are \
arbitrary, swapped=true means their best moment was transplanted from \
donorFrame, swapped=false means they were left as shot (fallbackReason says \
why). Be warm, concrete and brief; no technical jargon, no mentions of AI or \
models."""


def build_payload(report: dict) -> dict:
    """Reduce a Mily BestTakeReport to the anonymous numbers we send.

    Keeps: frameCount, baseFrame, and per person — label, baseFrame,
    donorFrame, swapped, checksPassed, fallbackReason, scores.
    Everything else in the report (per-frame score tables, candidate lists)
    stays on the device.
    """
    people = []
    for p in report.get("people", []):
        people.append({
            "label": p.get("person"),
            "baseFrame": p.get("baseFrame"),
            "donorFrame": p.get("donorFrame"),
            "swapped": p.get("swapped"),
            "checksPassed": p.get("checksPassed"),
            "fallbackReason": p.get("fallbackReason"),
            "baseScore": p.get("baseScore"),
            "donorScore": p.get("donorScore"),
        })
    return {
        "frameCount": report.get("frameCount"),
        "baseFrame": report.get("baseFrame"),
        "people": people,
    }


def build_messages(payload: dict) -> list[dict]:
    frame_part = (f"{payload['frameCount']} frames"
                  if payload.get("frameCount") else "several frames")
    user = (
        f"Burst: {frame_part}; base frame {payload.get('baseFrame')}.\n"
        f"Decisions (JSON): {json.dumps(payload, separators=(',', ':'))}\n\n"
        "Reply with EXACTLY two lines, nothing else, no markdown:\n"
        "WHY: 2-3 friendly sentences on why this photo is everyone's best "
        "take — who was lifted from a better moment, who was left as shot "
        "and why that was the right call.\n"
        "CAPTION: one warm share caption for the photo, at most 12 words."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def call_nemotron(messages: list[dict], api_key: str, model: str,
                  timeout: float = 45.0) -> str:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 400,
    }).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300]
        raise SystemExit(f"Nebius Token Factory returned HTTP {error.code}: {detail}")
    except urllib.error.URLError as error:
        raise SystemExit(f"could not reach {ENDPOINT}: {error.reason}")

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise SystemExit(f"unexpected response shape: {json.dumps(data)[:300]}")
    if not content:
        raise SystemExit("Nemotron returned no message content.")
    return content


def parse_reply(content: str) -> tuple[str, str]:
    """Find the WHY:/CAPTION: markers anywhere in the reply — reasoning
    models may think out loud first, and that preamble is ignored."""
    lowered = content.lower()
    why_at = lowered.find("why:")
    caption_at = lowered.find("caption:")
    if why_at == -1 or caption_at == -1:
        raise SystemExit(
            "could not find the WHY:/CAPTION: lines in the reply:\n"
            + content[:300])
    why = content[why_at + len("why:"):caption_at].strip()
    caption = content[caption_at + len("caption:"):].strip()
    return why, caption


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report", help="path to Mily's BestTakeReport JSON "
                        "(mily process <burst> --report report.json)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Token Factory model id (default {DEFAULT_MODEL})")
    args = parser.parse_args()

    api_key = os.environ.get("MILY_NEBIUS_API_KEY") or os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        print("no credential found. Set NEBIUS_API_KEY (or MILY_NEBIUS_API_KEY) to your")
        print("Nebius Token Factory API key: https://console.tokenfactory.nebius.com")
        sys.exit(1)

    try:
        with open(args.report, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read a Mily report from {args.report}: {error}")

    payload = build_payload(report)
    why, caption = parse_reply(call_nemotron(build_messages(payload), api_key, args.model))
    print(f"WHY: {why}")
    print(f"CAPTION: {caption}")


if __name__ == "__main__":
    main()
