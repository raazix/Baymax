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
- GET `/api/assistant/responses/{id}`: saved answer/citations/hash, without internal snapshot.
- POST `/api/assistant/responses/{id}/speech`: MP3 audio.
- POST `/api/assistant/transcribe`: raw audio body, WebM/Ogg/MP4/WAV/MP3, max 10 MiB.

Provider timeouts, denied keys, rate limits, malformed answers and unsupported audio produce readable errors without echoing provider payloads or credentials. Optional LINEGUARD_API_TOKEN also protects assistant endpoints; the current dashboard assumes local-demo access.

## Validation

88 backend tests passed, including mocked provider errors, unknown citations, snapshot tampering, read-only inspection behavior, and transcription review. NVIDIA explanation, ElevenLabs MP3 synthesis, and speech-to-text roundtrip passed with real providers; data/assistant-live-test.json is a local report. Real physical microphone interaction still needs testing on the user's browser.

References: [NVIDIA model card](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b/modelcard), [ElevenLabs speech](https://elevenlabs.io/docs/api-reference/text-to-speech/convert), [ElevenLabs transcription](https://elevenlabs.io/docs/api-reference/speech-to-text/convert).
