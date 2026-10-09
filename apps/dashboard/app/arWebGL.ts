import * as THREE from 'three';
import type { Quad } from './arTracking';
export function createAROverlay(canvas: HTMLCanvasElement) {
  // Some mobile drivers return a renderer before WebGL is fully usable, then
  // throw later from setPixelRatio/setSize/render. Treat the entire GL path as
  // optional so camera tracking and touch controls still start on those phones.
  let renderer: THREE.WebGLRenderer | null = null;
  let fallback: ReturnType<typeof createCanvasOverlay> | null = null;
  let activeLayer: HTMLCanvasElement | null = null;
  let texture: THREE.CanvasTexture | null = null;
  let geometry: THREE.PlaneGeometry | null = null;
  let material: THREE.MeshBasicMaterial | null = null;
  let lineGeometry: THREE.BufferGeometry | null = null;
  let lineMaterial: THREE.LineBasicMaterial | null = null;
  let scene: THREE.Scene | null = null;
  let camera: THREE.OrthographicCamera | null = null;
  let plane: THREE.Mesh | null = null;
  let outline: THREE.LineLoop | null = null;
  let display: CanvasRenderingContext2D | null = null;
  let width = 0, height = 0;
  const startFallback = (quad: Quad | null = null, edge: Quad | null = null, opacity = 0) => {
    try { texture?.dispose(); renderer?.dispose(); renderer?.forceContextLoss(); } catch { /* release best effort */ }
    renderer = null; texture = null;
    fallback ??= createCanvasOverlay(canvas);
    if (activeLayer) fallback.setLayer(activeLayer);
    fallback.render(quad, edge, opacity);
  };
  try {
    renderer = new THREE.WebGLRenderer({ canvas: document.createElement('canvas'), alpha: true, antialias: false, powerPreference: 'low-power' });
    display = canvas.getContext('2d');
    if (!display) throw new Error('2D overlay context unavailable');
    renderer.setClearColor(0x000000, 0); renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    scene = new THREE.Scene();
    camera = new THREE.OrthographicCamera(0, 1, 1, 0, .1, 10); camera.position.z = 1;
    geometry = new THREE.PlaneGeometry(1, 1);
    material = new THREE.MeshBasicMaterial({ transparent: true, depthTest: false, side: THREE.DoubleSide, toneMapped: false });
    plane = new THREE.Mesh(geometry, material); plane.visible = false; scene.add(plane);
    lineGeometry = new THREE.BufferGeometry().setFromPoints(Array.from({ length: 4 }, () => new THREE.Vector3()));
    lineMaterial = new THREE.LineBasicMaterial({ color: 0xa7f3cc, transparent: true, depthTest: false });
    outline = new THREE.LineLoop(lineGeometry, lineMaterial); outline.visible = false; scene.add(outline);
  } catch {
    startFallback();
  }
  const setLayer = (layer: HTMLCanvasElement) => {
    activeLayer = layer;
    if (fallback) { fallback.setLayer(layer); return; }
    try {
      texture?.dispose(); texture = new THREE.CanvasTexture(layer); texture.colorSpace = THREE.SRGBColorSpace;
      if (!material) throw new Error('WebGL material unavailable');
      material.map = texture; material.needsUpdate = true;
    } catch { startFallback(); }
  };
  const render = (quad: Quad | null, outlineQuad: Quad | null, opacity: number) => {
    if (fallback) { fallback.render(quad, outlineQuad, opacity); return; }
    try {
      if (!renderer || !scene || !camera || !geometry || !material || !plane || !lineGeometry || !outline) throw new Error('WebGL renderer unavailable');
      const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
      if (w !== width || h !== height) {
        width = w; height = h; renderer.setSize(w, h, false);
        canvas.width = w; canvas.height = h;
        camera.right = w; camera.top = h; camera.updateProjectionMatrix();
      }
      plane.visible = Boolean(quad && texture && opacity > 0);
      if (quad) {
        const positions = geometry.getAttribute('position');
        [0, 1, 3, 2].forEach((corner, index) => positions.setXYZ(index, quad[corner].x, h - quad[corner].y, 0));
        positions.needsUpdate = true; geometry.computeBoundingSphere(); material.opacity = opacity;
      }
      outline.visible = Boolean(outlineQuad);
      if (outlineQuad) {
        const positions = lineGeometry.getAttribute('position');
        outlineQuad.forEach((point, index) => positions.setXYZ(index, point.x, h - point.y, .01));
        positions.needsUpdate = true; lineGeometry.computeBoundingSphere();
      }
      renderer.render(scene, camera);
      display?.clearRect(0, 0, width, height); if (display) display.drawImage(renderer.domElement, 0, 0, width, height);
    } catch { startFallback(quad, outlineQuad, opacity); }
  };
  return { get mode() { return fallback ? 'Canvas fallback' : 'WebGL'; }, setLayer, render, dispose: () => {
    fallback?.dispose(); texture?.dispose(); geometry?.dispose(); material?.dispose(); lineGeometry?.dispose(); lineMaterial?.dispose();
    try { renderer?.dispose(); renderer?.forceContextLoss(); } catch { /* release best effort */ }
  } };
}

function createCanvasOverlay(canvas: HTMLCanvasElement) {
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Camera overlay could not create a 2D context.');
  let layer: HTMLCanvasElement | null = null;
  let pixelRatio = 1;
  const resize = () => {
    const width = Math.max(1, canvas.clientWidth), height = Math.max(1, canvas.clientHeight);
    pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    if (canvas.width !== Math.round(width * pixelRatio) || canvas.height !== Math.round(height * pixelRatio)) {
      canvas.width = Math.round(width * pixelRatio); canvas.height = Math.round(height * pixelRatio);
    }
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    return { width, height };
  };
  return {
    mode: 'Canvas fallback',
    setLayer: (image: HTMLCanvasElement) => { layer = image; },
    render: (quad: Quad | null, outline: Quad | null, opacity: number) => {
      const { width, height } = resize(); context.clearRect(0, 0, width, height);
      if (quad && layer && opacity > 0) {
        context.save(); context.globalAlpha = opacity;
        context.transform((quad[1].x - quad[0].x) / layer.width, (quad[1].y - quad[0].y) / layer.width,
          (quad[3].x - quad[0].x) / layer.height, (quad[3].y - quad[0].y) / layer.height, quad[0].x, quad[0].y);
        context.drawImage(layer, 0, 0); context.restore();
      }
      if (outline) {
        context.save(); context.beginPath(); context.moveTo(outline[0].x, outline[0].y);
        for (const point of outline.slice(1)) context.lineTo(point.x, point.y);
        context.closePath(); context.lineWidth = 3; context.strokeStyle = '#eaddff'; context.stroke(); context.restore();
      }
    },
    dispose: () => { layer = null; context.clearRect(0, 0, canvas.width, canvas.height); },
  };
}
