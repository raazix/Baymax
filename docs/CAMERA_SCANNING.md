# Camera scanning

On the inspection dashboard, choose **Scan with camera** beside image upload. The scanner opens a live preview using the rear camera where available. Keep the entire part visible and choose **Capture and inspect**. It captures a full-resolution PNG and runs the same inspection workflow as uploaded images, using the currently selected model. The result opens in the inspection console and is saved to history. Use **Switch camera** to request the other facing direction on supported devices.

Camera access requires localhost or HTTPS and browser permission. The preview requests 30 fps and performs no OpenCV processing. Closing the scanner stops its stream and aborts its pending request; camera switches stop the previous stream. Captures are submitted once per click, with duplicate clicks prevented while processing.

The request uses `/api/inspections/upload?input_source=camera`. Camera scans use the `camera` quality profile, save `context.input_source=camera`, and record `actor=camera_scan` in the creation audit entry. Failed-quality captures are saved with recapture disposition and do not run inference. The existing `source=uploaded_image` transport classification remains for response compatibility; context and audit identify camera acquisition. File uploads default to `input_source=upload` and retain their model-specific proxy quality profile.

The dashboard labels camera-acquired results as camera inspections. Quality, model inference, deterministic triage, analytics, recommendations, evidence and optional voice alerts use the existing workflow. Current models remain proxies; process inputs remain simulated unless real telemetry is supplied elsewhere.

Validation: Next.js production build and TypeScript checks, 10 backend tests, plus an actual API camera-source request on a sample PNG passed. The API check confirmed full inspection/analytics/action output, camera quality profile, camera audit and persisted image/hash. Physical camera permission and device switching need testing on the user's device.
