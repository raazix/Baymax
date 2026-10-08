# Markerless bottle overlay demo

The dashboard's bottle AR demo uses OpenCV.js in the browser to find a tall, centered contour in the webcam image. It draws a smoothed outline and the inspection's PatchCore grid over that screen-space region. It needs no printed marker.

## Try it locally

1. Start the API and dashboard as usual and open an uploaded casting inspection that has a PatchCore heatmap.
2. Choose **Try bottle AR demo** and allow webcam access. Camera access requires localhost or HTTPS.
3. Place a steel bottle upright in the center of the camera view. Use a plain contrasting background, show the full bottle, and keep the camera steady.

## Scope and limits

This is a lightweight markerless visualization prototype. OpenCV contour geometry estimates a 2D bounding region; it does not recognize bottles, estimate full 6-DoF pose, or reconstruct the curved metal surface. Reflections, weak contrast, clutter, camera motion, and partial occlusion can cause the outline to drift or disappear. The heatmap is drawn inside the detected outline but comes from a separate inspection image, so it is illustrative and is not registered to actual bottle defects. It is not validated for brake discs, bottle inspection, acceptance decisions, metrology, or safety determinations.

The full-frame OpenCV.js runtime is served locally from the installed npm package at `/api/opencvjs`, so the browser does not need to fetch a third-party runtime CDN.

## Validation status

TypeScript and the Next.js production build pass. The OpenCV runtime route responds with JavaScript. Webcam permission, contour lock, and overlay behavior still need a live browser/device check with the bottle.
