# Responsive workspace and phone demo

Updated 2026-10-09. The familiar industrial UI keeps its graphite/teal palette and IBM Plex Sans. The image leads the desktop layout, with cause, risk and action grouped beside it. Details remain expandable. Phone layouts use a single evidence column, bottom navigation, touch-sized controls and full-screen camera capture.

## Motion and effects

- Motion: shared navigation selection, interruptible tab/scene transitions, animated forecast values, alert appearance and touch feedback.
- Three.js: production lifecycle and optional rotor/heatmap views. Heavy 3D modules load when opened. Scene rendering stops while hidden/offscreen and uses a smaller phone pixel/frame budget.
- [Radix Dialog](https://www.radix-ui.com/primitives/docs/components/dialog): focus-contained camera, pairing and tool sheets.
- [Sonner](https://github.com/emilkowalski/sonner): processing, saved-result, error and critical-hold notifications.
- [qrcode.react](https://github.com/zpao/qrcode.react): QR pairing for the real phone URL.
- Lucide: consistent control/status icons. Reduced-motion preferences remain supported.

Critical findings still require engineer approval and explicit resume. An unknown anomaly does not become a critical crack because of an animation or an LLM response. Proxy/synthetic limits stay visible.

## Open it on a real phone

Start FastAPI normally, then run from the repository root:

```powershell
powershell -File scripts/start_phone_demo.ps1
```

If the current production build already exists, add `-SkipBuild`. The script starts an additional local dashboard on port 3002 and a [Cloudflare Quick Tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/), then prints an HTTPS URL and six-digit pairing code. It leaves the desktop server and FastAPI running.

On the laptop, **Connect phone** shows the QR and code. On the phone:

1. Open the HTTPS URL in Chrome or Safari, or scan the QR.
2. Enter the code. The phone receives a signed, Secure, HttpOnly session cookie valid for eight hours.
3. Tap **Scan** and allow camera access, then **Capture and inspect**. The result uses the laptop's models/database. **Use the phone camera app** is an alternative image-capture path.
4. Tap **Show live line** to view normal production animation, then **Demonstrate critical defect** for an explicitly synthetic critical scenario. Approve and explicitly resume to show the response workflow.
5. Use **Sensors**, **Evidence**, and **More** for the other workspace tools. Audio controls remain if the browser blocks automatic playback.

The tunnel supports a phone on another network too. Keep the laptop awake, FastAPI running and the tunnel active. Restarting the tunnel generates a new link. This is a temporary demo, not a permanent deployment.

```powershell
powershell -File scripts/stop_phone_demo.ps1
```

The stop script verifies the saved process command lines and stops only the phone dashboard/tunnel. Pairing state and logs live in ignored `data/`; session secrets are passed only to the phone server environment. Unpaired API access is blocked, incorrect codes are rejected, and repeated attempts are rate-limited. This pairing gate does not turn asserted engineer names into authenticated production identities.

## Verification evidence

- Next.js production build and TypeScript passed.
- Desktop 1440px, Pixel 5 dimensions, iPhone 13 dimensions and a 320px phone layout checked without horizontal overflow.
- A generated sharp WebRTC frame passed the camera quality gate, actual PatchCore/process-analytics path and persisted camera provenance.
- Mobile sensor/lab navigation and critical hold -> engineer approval -> explicit resume passed with no browser runtime errors.
- Public HTTPS checks passed: clean phone redirect, unpaired API 401, bad-code rejection, secure cookie, paired dashboard/API access and QR link.
- A mobile browser paired over the actual public HTTPS tunnel, obtained a secure camera context and completed a generated camera capture through the real backend.
- Session helpers checked code matching, Unicode rejection, expiry and tampering. The design detector reported no findings; desktop/phone/camera screenshots were inspected in two bounded passes.

Phone browser tests use generated camera imagery and Chromium emulation. They do **not** establish physical phone camera performance, Safari-engine behavior, real defect accuracy or guaranteed speaker autoplay. Confirm permissions, lighting and audio on the user's device.

Regression scripts:

```powershell
node scripts/test_mobile_dashboard.cjs
node scripts/test_production_line.cjs
```

These require Playwright and its Chromium (or `PLAYWRIGHT_EXECUTABLE`) and create demo inspection records. The mobile script generates `data/browser-check/phone-camera.y4m` if missing. Local screenshots and verification metadata are saved under ignored `data/browser-check/`.

The remaining model/history/batch-alert requirements in `docs/TRACK3.md` are unchanged by this UI work.
