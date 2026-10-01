# Best-take explainer — NVIDIA Nemotron on Nebius Token Factory

The hackathon piece of [Mily](https://github.com/firebreathingunicorn/mily)
(Nebius x NVIDIA Global AI Hackathon, **Best Apps and Agents track**): a
tiny, dependency-free service that turns Mily's on-device decision report
into two human sentences — *why this photo is everyone's best take*, and a
share caption — using **NVIDIA Nemotron 3 Super** served on **Nebius Token
Factory**.

```
Mily (on-device, never uploads pixels)          Nebius Token Factory
─────────────────────────────────────           ─────────────────────────
mily process <burst> --report report.json
        │
        ▼  anonymous numbers only
   report.json  ──────────────────────►  POST /v1/chat/completions
   frame indices, check outcomes,              │  nvidia/nemotron-3-super-120b-a12b
   scores — no images, no faces                ▼
        ▲                             WHY:  everyone's best moment…
        └────────────────────────    CAPTION: one photo, everyone shining
```

## Why this split

Mily's engine is deliberately on-device: Vision-based scoring, transplant
synthesis, artifact + identity checks — nothing uploads. The one thing a
photo app can't do offline is *words*: explain the decision like a friend
would, and write the caption you'd actually post. That's the part Nemotron
does, on Nebius infrastructure, from the report the engine already writes.

**What leaves the device:** `frameCount`, `baseFrame`, and per person —
label, `baseFrame`, `donorFrame`, `swapped`, `checksPassed`,
`fallbackReason`, two scores. That is the complete payload (enforced by
`build_payload`, unit-tested). No pixels, no faces, no metadata.

## Setup

```bash
export NEBIUS_API_KEY=...   # https://console.tokenfactory.nebius.com
python3 test_explainer.py   # 9 unit tests, no network, no key needed
python3 explainer.py example/report.example.json
```

With a real Mily report:

```bash
cd Mily && swift run mily process <burst-dir> --report /tmp/report.json
python3 explainer.py /tmp/report.json
```

Output:

```
WHY: B was mid-blink in the base frame, so Mily lifted their smile from
frame 4 where the checks passed cleanly. A stayed exactly as shot — every
candidate frame caught a blink, and a real photo beats a risky edit.
CAPTION: everyone's best moment, one photo
```

## Model tiers

| Tier | Model id | Why pick it |
|---|---|---|
| Super (default) | `nvidia/nemotron-3-super-120b-a12b` | fast everyday calls (256K context) — this job is structured-JSON → friendly text |
| Ultra | `nvidia/Nemotron-3-Ultra-550b-a55b` | the 1M-context reasoner, if you want the app's "photo story" feature later |

`python3 explainer.py report.json --model nvidia/Nemotron-3-Ultra-550b-a55b`

The endpoint is OpenAI-compatible
(`https://api.tokenfactory.nebius.com/v1/chat/completions`, `Authorization:
Bearer $NEBIUS_API_KEY`), so the same payload works from any OpenAI SDK —
this repo just uses stdlib `urllib` so there is nothing to install.

## Cost

One explanation is ~0.9k tokens (system + report JSON + ≤400 out). At
Nebius's blended **$0.36 / 1M tokens** for Super (Artificial Analysis,
Oct 2026): **≈ $0.0003 per photo — about $3 per 10,000 explanations.**

## Files

```
explainer.py                  the whole service: payload → prompt → Token Factory → WHY/CAPTION
test_explainer.py             stdlib unit tests: payload privacy, marker parsing, wire shape
example/report.example.json   a real-shaped Mily BestTakeReport to try it with
```

## License

MIT — see [LICENSE](LICENSE).
