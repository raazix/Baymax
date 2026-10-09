# Inspection assistant and voice

The local backend loads `.env` from the project root without overriding existing environment variables. Never commit this file or use NEXT_PUBLIC variables for provider keys. Rotate credentials shared in chat.

## Models

Default NVIDIA model: `nvidia/nemotron-3-ultra-550b-a55b`, available in the authenticated catalog and tested successfully. This is a high-capacity NVIDIA-native choice, not a proven best model for LineGuard or an additional accuracy claim. Reasoning is disabled for concise answers; temperature 0.2, max 800 tokens. Configure NVIDIA_MODEL to change it explicitly.

ElevenLabs uses `eleven_flash_v2_5` for low-latency speech, Sarah as the default voice, and `scribe_v2` for transcription. Change ELEVENLABS_VOICE_ID / ELEVENLABS_TTS_MODEL on the backend.

## Dashboard

Select an inspection and find Inspection assistant below the decision panels. Type a question or click the microphone. Recording stops on demand or at 30 seconds; check/edit the transcript, then click Explain. Microphone questions request a spoken answer automatically; text answers can be read using Listen. Browser autoplay may require pressing Play in the audio control. Microphone access requires localhost or HTTPS and browser permission. Recording tracks stop when the component closes; blob URLs are revoked.

Requests transmit the selected inspection evidence and question to NVIDIA, recorded audio to ElevenLabs for transcription, and the generated answer to ElevenLabs for speech. The keys never reach the browser. A question is sent only on Explain; recording uploads on Stop.

## Evidence and boundaries

Read-only explanations use the selected inspection's stored quality, defects, anomaly score/threshold, calibration, process context, analytics and recorded SOP. Prompted constraints preserve proxy/synthetic labels, unknown causes and engineer approval. Responses use validated citation IDs, schema bounds, and a saved evidence snapshot; prompts cannot grant approval tools because none exist. Model-generated prose can still be wrong and requires engineer review. No LLM-produced severity, forecasts, measurements or control commands become authoritative records.

Responses persist under ignored `data/assistant/<uuid>.json` with an evidence digest and record hash. Speech accepts only a saved response ID, not arbitrary client text. Tampered response files fail integrity checks. Original inspections and their audit chains remain unchanged. Audio clips/transcripts are not persisted by the backend; upstream provider retention follows the account's provider settings.

## API

- GET `/api/assistant/status`: configuration flags and model IDs; no keys.
- POST `/api/inspections/{id}/assistant`: JSON question (max 1,200 characters).
- POST `/api/inspections/{id}/vision-description`: reads the exact saved, quality-passed upload; verifies SHA-256; returns a tentative NVIDIA vision description tied to the image hash. It cannot set severity, disposition or approval. AR Live AI Vision samples the latest completed crop approximately every 10 seconds; local CV/model tracking continues between calls. Default vision model: `meta/llama-3.2-11b-vision-instruct` (`NVIDIA_VISION_MODEL` overrides it).
- GET `/api/assistant/responses/{id}`: saved answer/citations/hash, without internal snapshot.
- POST `/api/assistant/responses/{id}/speech`: MP3 audio.
- POST `/api/assistant/transcribe`: raw audio body, WebM/Ogg/MP4/WAV/MP3, max 10 MiB.

Provider timeouts, denied keys, rate limits, malformed answers and unsupported audio produce readable errors without echoing provider payloads or credentials. Optional LINEGUARD_API_TOKEN also protects assistant endpoints; the current dashboard assumes local-demo access.

## Validation

88 backend tests passed, including mocked provider errors, unknown citations, snapshot tampering, read-only inspection behavior, and transcription review. NVIDIA explanation, ElevenLabs MP3 synthesis, and speech-to-text roundtrip passed with real providers; data/assistant-live-test.json is a local report. Real physical microphone interaction still needs testing on the user's browser.

References: [NVIDIA Llama 3.2 Vision API](https://docs.nvidia.com/nim/vision-language-models/1.2.0/examples/llama3-2/api.html), [NVIDIA model card](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b/modelcard), [ElevenLabs speech](https://elevenlabs.io/docs/api-reference/text-to-speech/convert), [ElevenLabs transcription](https://elevenlabs.io/docs/api-reference/speech-to-text/convert).

## LLM layer tuning (2026-10-09)

The hosted Nemotron models cannot be weight-fine-tuned through the NVIDIA API (and a 120B-550B model cannot be trained
on a 6 GB laptop GPU), so tuning was done on the inputs, checks and model choice, measured with `scripts/eval_assistant.py`.

- **Bug fixed:** `.env` starts with a UTF-8 byte-order mark, so the first key was read as `﻿NVIDIA_API_KEY` and the
  assistant reported `llm_configured: false`. `core/config.py` now loads `.env` with `encoding='utf-8-sig'`.
- **Compact evidence packet** (`compact_packet`): about 2,600 characters instead of about 9,600; keeps findings, severity
  factors, lot drift, RCA top drivers, forecast, action and provenance flags with the same citation IDs.
- **Number grounding:** every number in an answer must appear in the evidence (fractions may be quoted as percentages,
  ranges are parsed). One corrective retry names the offending numbers; otherwise the answer is rejected with the reason.
- **Citation aliases:** section names such as `forecast` are mapped to their citation ID.
- **Answer cache:** keyed on model, normalised question and evidence hash (`data/assistant/cache_index.json`).
- **Model choice:** automatic summaries send `purpose: summary`.

| Configuration (3 inspections x 3 questions) | Grounded | Median latency | Notes |
|---|---|---|---|
| Ultra 550B, full packet | 9/9 | 9.7 s | 3.7x the prompt size |
| Ultra 550B, compact | 5/9 | 7.4 s | all 4 failures were NVIDIA HTTP 503 |
| **Super 120B, compact** | **9/9** | **5.0 s** | default for summaries and questions |
| Lightning 30B, compact | 2/9 | 11.7 s | replies not valid JSON; not used |

Super is the default (`NVIDIA_MODEL` / `NVIDIA_SUMMARY_MODEL` override it). On a provider 5xx or timeout the request
falls back once to Ultra (`NVIDIA_FALLBACK_MODEL`). Reports: `data/assistant_eval_round1.json`, `data/assistant_eval_round2.json`.
The grounding check verifies numbers and citations, not every claim; engineer review remains required.
