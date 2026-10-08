'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

export type RotorDefect = { label: string; r_mm: number; theta_deg: number; equivalent_diameter_mm: number; zone: string; severity: { level: string } };
export type RotorHeatmap = { grid: number[][]; threshold: number };

const SEVERITY_COLOR: Record<string, number> = { critical: 0xd8342a, high: 0xe0672b, medium: 0xe0a82b, low: 0x3b7bb5 };
const ZONES = [
  { name: 'Mounting hat', from: 26, to: 45, color: 0x3b7bb5 },
  { name: 'Vane root', from: 45, to: 90, color: 0xe0a82b },
  { name: 'Friction face', from: 90, to: 135, color: 0x2f9e7a },
];
const OUTER_R = 135;
const HAT_Y = 34;
const FACE_Y = 22;

// Top-surface height (mm) at radius r, matching the lathe profile below.
function surfaceY(r: number) {
  if (r <= 48) return HAT_Y;
  if (r >= 56) return FACE_Y;
  return HAT_Y + ((r - 48) / 8) * (FACE_Y - HAT_Y);
}

// Image axes (x right, y down) map to world (x, z) so the 3D top view matches the 2D inspection image.
function polarToWorld(r: number, thetaDeg: number, lift = 0) {
  const t = THREE.MathUtils.degToRad(thetaDeg);
  return new THREE.Vector3(r * Math.cos(t), surfaceY(r) + lift, r * Math.sin(t));
}

function buildRotor() {
  const group = new THREE.Group();
  const profile = [[135, 22], [135, 0], [88, 0], [56, 10], [48, 26], [26, 26], [26, 34], [48, 34], [56, 22], [135, 22]]
    .map(([r, y]) => new THREE.Vector2(r, y));
  const body = new THREE.Mesh(
    new THREE.LatheGeometry(profile, 160),
    new THREE.MeshStandardMaterial({ color: 0x9aa3a6, metalness: 0.85, roughness: 0.38, side: THREE.DoubleSide }),
  );
  group.add(body);

  // Machined friction-face grooves and bolt holes (visual detail only).
  const grooveMat = new THREE.MeshBasicMaterial({ color: 0x5d6669, side: THREE.DoubleSide });
  for (const r of [100, 112, 124]) {
    const groove = new THREE.Mesh(new THREE.RingGeometry(r, r + 0.5, 160), grooveMat);
    groove.rotation.x = -Math.PI / 2;
    groove.position.y = FACE_Y + 0.05;
    group.add(groove);
  }
  const boltMat = new THREE.MeshBasicMaterial({ color: 0x15191a });
  for (let i = 0; i < 5; i++) {
    const hole = new THREE.Mesh(new THREE.CircleGeometry(3.6, 32), boltMat);
    const a = (i / 5) * Math.PI * 2;
    hole.rotation.x = -Math.PI / 2;
    hole.position.set(37 * Math.cos(a), HAT_Y + 0.06, 37 * Math.sin(a));
    group.add(hole);
  }
  const bore = new THREE.Mesh(new THREE.CircleGeometry(26, 64), new THREE.MeshBasicMaterial({ color: 0x101415 }));
  bore.rotation.x = -Math.PI / 2;
  bore.position.y = HAT_Y + 0.06;
  group.add(bore);
  return group;
}

function buildZones() {
  const group = new THREE.Group();
  for (const zone of ZONES) {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(zone.from, zone.to, 128),
      new THREE.MeshBasicMaterial({ color: zone.color, transparent: true, opacity: 0.4, side: THREE.DoubleSide, depthWrite: false }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = surfaceY((zone.from + zone.to) / 2) + 0.3;
    group.add(ring);
  }
  group.visible = false;
  return group;
}

// Project image-space PatchCore distances onto the procedural rotor as a visual aid.
// This deliberately does not claim image-to-part registration or pixel segmentation.
function buildHeatmap(heatmap: RotorHeatmap) {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx || heatmap.grid.length === 0 || heatmap.grid[0]?.length === 0) return null;
  const values = heatmap.grid.flat();
  const min = Math.min(...values);
  const max = Math.max(...values);
  const belowRange = Math.max(heatmap.threshold - min, 1e-6);
  const aboveRange = Math.max(max - heatmap.threshold, 1e-6);
  const rows = heatmap.grid.length;
  const cols = heatmap.grid[0].length;
  ctx.save();
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
  ctx.clip();
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const value = heatmap.grid[row][col];
      const aboveThreshold = value > heatmap.threshold;
      const normalized = aboveThreshold
        ? Math.min(1, (value - heatmap.threshold) / aboveRange)
        : Math.min(1, Math.max(0, (value - min) / belowRange));
      const hue = aboveThreshold ? 55 - normalized * 55 : 215 - normalized * 25;
      const alpha = aboveThreshold ? 0.3 + normalized * 0.55 : 0.12 + normalized * 0.12;
      ctx.fillStyle = `hsla(${hue}, 92%, 53%, ${alpha})`;
      ctx.fillRect(col * size / cols, row * size / rows, size / cols + 1, size / rows + 1);
    }
  }
  ctx.restore();
  // Keep the modeled hub clear; this is a rotor-shaped projection for presentation.
  ctx.globalCompositeOperation = 'destination-out';
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, (56 / OUTER_R) * size / 2, 0, Math.PI * 2);
  ctx.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(
    new THREE.CircleGeometry(OUTER_R, 128),
    new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2 }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = FACE_Y + 0.18;
  mesh.userData.heatmapTexture = texture;
  return mesh;
}

function buildMarker(defect: RotorDefect) {
  const color = SEVERITY_COLOR[defect.severity.level] ?? 0xd8342a;
  const group = new THREE.Group();
  const size = Math.max(defect.equivalent_diameter_mm, 5);
  const pin = 34;
  const base = polarToWorld(defect.r_mm, defect.theta_deg);

  const spot = new THREE.Mesh(new THREE.CircleGeometry(size, 40), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }));
  spot.rotation.x = -Math.PI / 2;
  spot.position.copy(base).y += 0.5;
  const pulse = new THREE.Mesh(new THREE.RingGeometry(size * 1.1, size * 1.35, 48), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthWrite: false }));
  pulse.rotation.x = -Math.PI / 2;
  pulse.position.copy(spot.position);
  pulse.userData.pulse = true;

  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([base.clone().setY(base.y + 0.5), base.clone().setY(base.y + pin)]),
    new THREE.LineBasicMaterial({ color }),
  );
  const head = new THREE.Mesh(new THREE.SphereGeometry(4.2, 24, 16), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.9 }));
  head.position.copy(base).y += pin;
  group.add(spot, pulse, line, head);
  return group;
}

function disposeTree(root: THREE.Object3D) {
  root.traverse(obj => {
    const mesh = obj as THREE.Mesh;
    mesh.geometry?.dispose();
    const material = mesh.material;
    if (Array.isArray(material)) material.forEach(m => m.dispose());
    else material?.dispose();
  });
}

export default function RotorViewer({ defects, heatmap }: { defects: RotorDefect[]; heatmap?: RotorHeatmap | null }) {
  const mount = useRef<HTMLDivElement>(null);
  const markers = useRef<THREE.Group | null>(null);
  const zones = useRef<THREE.Group | null>(null);
  const heatLayer = useRef<THREE.Group | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const [autoRotate, setAutoRotate] = useState(false);
  const [showZones, setShowZones] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [unsupported, setUnsupported] = useState(false);
  const flags = useRef({ autoRotate: false });

  // Scene setup once.
  useEffect(() => {
    const el = mount.current;
    if (!el) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      setUnsupported(true);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x1b2a30);
    el.appendChild(renderer.domElement);
    renderer.domElement.setAttribute('aria-hidden', 'true');

    const scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(renderer);
    const envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = envTexture;
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(120, 220, 90);
    scene.add(sun);

    const camera = new THREE.PerspectiveCamera(38, 1, 1, 2000);
    const home = new THREE.Vector3(0, 330, 250);
    camera.position.copy(home);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.minDistance = 120;
    controls.maxDistance = 650;
    controls.maxPolarAngle = Math.PI * 0.62;
    controls.autoRotateSpeed = 1.2;
    controlsRef.current = controls;

    const rotor = buildRotor();
    const zoneGroup = buildZones();
    const markerGroup = new THREE.Group();
    const heatGroup = new THREE.Group();
    scene.add(rotor, zoneGroup, markerGroup, heatGroup);
    markers.current = markerGroup;
    zones.current = zoneGroup;
    heatLayer.current = heatGroup;
    const floor = new THREE.GridHelper(420, 14, 0x3c545c, 0x2a3d44);
    floor.position.y = -1;
    scene.add(floor);

    const resize = () => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(el);

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let frame = 0;
    const start = performance.now();
    const loop = () => {
      frame = requestAnimationFrame(loop);
      controls.autoRotate = flags.current.autoRotate && !reduceMotion;
      controls.update();
      const t = (performance.now() - start) / 1000;
      markerGroup.traverse(obj => {
        if (obj.userData.pulse && !reduceMotion) {
          const s = 1 + 0.25 * Math.sin(t * 3);
          obj.scale.set(s, s, s);
        }
      });
      renderer.render(scene, camera);
    };
    loop();
    (el as HTMLDivElement & { resetView?: () => void }).resetView = () => {
      camera.position.copy(home);
      controls.target.set(0, 0, 0);
    };

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      disposeTree(scene);
      envTexture.dispose();
      pmrem.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      markers.current = null;
      zones.current = null;
      heatLayer.current = null;
      controlsRef.current = null;
    };
  }, []);

  useEffect(() => {
    const group = heatLayer.current;
    if (!group) return;
    for (const child of [...group.children]) {
      group.remove(child);
      const texture = child.userData.heatmapTexture as THREE.Texture | undefined;
      disposeTree(child);
      texture?.dispose();
    }
    const layer = heatmap ? buildHeatmap(heatmap) : null;
    if (layer) group.add(layer);
    group.visible = Boolean(layer && showHeatmap);
  }, [heatmap, showHeatmap]);

  // Rebuild defect markers when the inspection changes.
  useEffect(() => {
    const group = markers.current;
    if (!group) return;
    for (const child of [...group.children]) {
      group.remove(child);
      disposeTree(child);
    }
    for (const defect of defects) group.add(buildMarker(defect));
  }, [defects]);

  useEffect(() => { flags.current.autoRotate = autoRotate; }, [autoRotate]);
  useEffect(() => { if (zones.current) zones.current.visible = showZones; }, [showZones]);

  const summary = defects.length
    ? defects.map(d => `${d.label.replaceAll('_', ' ')} at ${d.r_mm} mm, ${d.theta_deg}°`).join('; ')
    : 'No defects detected';

  return <div className="rotor-viewer">
    <div className="rotor-canvas" ref={mount} role="img" aria-label={`3D rotor model. ${summary}.`}>
      {unsupported && <p className="rotor-fallback">WebGL is unavailable in this browser, so the 3D view cannot be shown.</p>}
    </div>
    <div className="rotor-side">
      <div className="buttons">
        <button onClick={() => setAutoRotate(v => !v)} aria-pressed={autoRotate}>{autoRotate ? 'Stop rotation' : 'Auto-rotate'}</button>
        <button onClick={() => setShowZones(v => !v)} aria-pressed={showZones}>{showZones ? 'Hide zones' : 'Show zones'}</button>
        {heatmap && <button onClick={() => setShowHeatmap(v => !v)} aria-pressed={showHeatmap}>{showHeatmap ? 'Hide heatmap' : 'Show heatmap'}</button>}
        <button onClick={() => { (mount.current as (HTMLDivElement & { resetView?: () => void }) | null)?.resetView?.(); }}>Reset view</button>
      </div>
      {showZones && <ul className="zone-legend">{ZONES.map(z => <li key={z.name}><i style={{ background: `#${z.color.toString(16).padStart(6, '0')}` }} />{z.name} · {z.from}–{z.to} mm</li>)}</ul>}
      <h3>Located defects</h3>
      {defects.length === 0 ? <p className="muted">No defects to place on this part.</p> : <ul className="defect-list">{defects.map((d, i) => <li key={i}><i style={{ background: `#${(SEVERITY_COLOR[d.severity.level] ?? 0xd8342a).toString(16).padStart(6, '0')}` }} /><div><strong>{d.label.replaceAll('_', ' ')}</strong><span>{d.severity.level} · r {d.r_mm} mm · θ {d.theta_deg}° · {d.zone.replaceAll('_', ' ')}</span></div></li>)}</ul>}
      <p className="footnote">Drag to orbit, scroll to zoom. {heatmap ? 'The PatchCore distance grid is projected onto a procedural rotor for visualization; it is not a pixel mask, physical registration, or camera-tracked AR.' : 'Procedural rotor with nominal proportions. Defect positions come from replay measurements.'}</p>
    </div>
  </div>;
}
