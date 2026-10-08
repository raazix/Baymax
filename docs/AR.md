# Marker-tracked AR preview

LineGuard now has a live camera AR view using AR.js marker pose tracking and Three.js rendering.

## Try it locally

1. Open an uploaded PatchCore inspection in the dashboard.
2. Open `/marker` in a separate tab and print at 100% with fit-to-page disabled. The Hiro marker's outer black square should measure 40 mm.
3. Put the marker flat at the center of the inspected rotor hub, facing the camera. Do not move it during the AR view.
4. Choose **Open tracked AR**, allow camera access, and hold the webcam so the marker and rotor are visible.
5. Enter the rotor's measured outside diameter. Adjust image rotation and X/Z center offsets only to visually align the overlay.

Camera access requires `localhost` or HTTPS. The normal local dashboard uses `127.0.0.1:3000`, which browsers treat as a secure context.

## What is tracked and what is calibrated

AR.js estimates the Hiro marker's 6-DoF pose from the webcam and Three.js attaches the translucent overlay to that moving marker. The marker must remain rigidly related to the part. The 40 mm marker size is configured from its measured printed black-square width. Rotor outside diameter and marker-to-image center/rotation alignment are user-entered; they are not automatically measured or validated. The overlay remains planar and is not a volumetric surface scan.

The displayed colors come from the PatchCore spatial distance grid, with the configured threshold separating blue from amber/red. A PatchCore grid is not a pixel-accurate segmentation mask. It comes from the uploaded inspection and only aligns with the physical part if the captured image shows the same part in the same orientation and the operator calibrates the center and rotation. Current default image/model is an MPDD metal-plate proxy; it is not brake-disc validated. Do not use this preview for acceptance decisions, safety determinations, or metrology.

## Validation status

Next.js production build and TypeScript checks pass. Marker page and static pattern/camera calibration assets return HTTP 200 on the local dashboard; production dependency audit reports zero vulnerabilities after pinning Axios 1.20.0 for the AR.js dependency tree. Physical camera permission, marker lock, pose stability, and overlay alignment still require testing on the target webcam/device. AR.js's marker controls set the live marker pose; the uploaded heatmap alignment is a separate manual calibration step.

AR.js marker concepts and its Three.js API are described in the [official marker tracking documentation](https://ar-js-org.github.io/AR.js-Docs/marker-based/). Browser camera APIs require a secure context; see [MDN's WebXR and secure-context overview](https://developer.mozilla.org/en-US/docs/Web/API/WebXR_Device_API).
