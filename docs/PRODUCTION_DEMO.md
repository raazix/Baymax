# Production lifecycle and ElevenLabs alerts

The dashboard includes a procedural Three.js production lifecycle: casting, machining, inspection, engineer review and dispatch. It is an illustrative simulation, not a connected factory digital twin.

## Try it

1. Start FastAPI and the Next.js dashboard as described in README. Open the inspection console. Local verification preview: `http://127.0.0.1:3001` (port 3000 was already occupied).
2. Click **Demonstrate critical defect**. This creates an explicitly synthetic thermal-drift rotor inspection through `/api/replay`, with stored evidence, trained synthetic analytics and a critical crack severity fixture. It does not pretend to detect a crack in a real bottle image.
3. The conveyor stops, the inspection station and representative disc turn red, and a visible critical notification identifies part, lot and machine. ElevenLabs generates spoken stored findings and the engineering recommendation.
4. Enter the engineer name and approve the action. The conveyor stays stopped until **Resume simulation after approval** is clicked. Reject/escalate do not enable resume. A normal inspection does not clear a hold. Multiple observed critical inspections wait for individual approval/resume.
5. Upload an image or scan with a camera: results feed the same critical-hold logic. Current proxy anomaly models generally return `review_required`, not `critical`, because an unknown anomaly alone does not establish physical safety severity.

## Voice behavior

- Uses the backend `ELEVENLABS_API_KEY`, optional `ELEVENLABS_VOICE_ID` and `ELEVENLABS_TTS_MODEL`. No credentials enter client code or Git.
- `/api/inspections/{id}/alert-speech` reads persisted high/critical findings and SOP text. No LLM sets severity, approves actions or invents a measurement for the alert.
- Rejected captures/normal findings return 409 before calling the provider.
- Synthetic/proxy evidence scope is spoken explicitly. The audio does not claim a real machine stopped.
- The dashboard uses one audio player, deduplicates automatic alerts per inspection and aborts superseded requests. Replay voice is available in the hold banner.
- Browsers may block autoplay; visible controls and a playback notice remain. Provider failures appear visibly and do not clear the hold.
- Voice also reads high findings; the simulated production hold is specifically triggered by critical findings.

## Performance and verification

The 3D loop is capped near 30 fps, caps pixel ratio at 1.5, pauses hidden-tab rendering, avoids continuous rendering when held/paused, respects reduced-motion preferences and disposes GPU resources on unmount. The line represents inspection response, not tracking of the actual photographed part through manufacturing.

Checks for this change:

- Assistant tests: 8 passed, including alert text scope, immutable evidence, rejected/normal gating and provider use.
- Backend tests: 11 passed, including actual alert endpoint returning mocked ElevenLabs audio and blocking normal alerts.
- Next.js production build/type checks passed.
- Live PostgreSQL-backed critical replay + ElevenLabs speech returned HTTP 200, `audio/mpeg`, 459,381 bytes in the direct smoke test.
- Browser workflow exercised real API replay/approval, a WebGL canvas, critical notification, approval interlock, explicit resume and pause/play without runtime errors. Optional audio autoplay depends on browser policy.
- Next.js now uses `app/api/[...path]/route.ts` for explicit byte forwarding instead of the rewrite proxy that intermittently stalled requests. It preserves query strings/content types/authorization, streams responses, limits bodies to 11 MiB, allows 120 seconds for ordinary API operations and 10 minutes for custom training, and never automatically retries state-changing requests. Backend-specific smaller limits still apply.
- Forwarding smoke passed: PostgreSQL health, raw camera PNG and camera provenance, exact returned image bytes, invalid-query 422, and oversized-body 413. Local `/api/opencvjs` remains a separate static asset handler.

Browser regression (creates synthetic demo records and uses configured voice provider):

```powershell
npm.cmd install --prefix data/browser-tools --no-save playwright
$env:DASHBOARD_URL = 'http://127.0.0.1:3001'
node scripts/test_production_line.cjs
```

Install the matching Playwright Chromium if needed, or set `PLAYWRIGHT_EXECUTABLE` to a cached Chromium path. Set `SAVE_SCREENSHOT=1` for an optional full-page screenshot. Dataset/model binaries and generated screenshots/audio remain local and excluded from Git.

See `docs/TRACK3.md` for the exact PPT requirements and remaining gaps. This animation improves the demo/storytelling requirement; it does not replace component validation, historical prediction or the required grouped batch-alert summary.
