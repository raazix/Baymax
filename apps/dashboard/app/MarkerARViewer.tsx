'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { X } from 'lucide-react';
import { buildHeatmap, type RotorHeatmap } from './RotorViewer';

type ToolkitSource = { domElement: HTMLVideoElement | null; ready: boolean; init: (ready: () => void, error?: (e: { name?: string; message?: string }) => void) => void; onResizeElement: () => void; copyElementSizeTo: (el: HTMLElement) => void; dispose: () => void };
type ToolkitContext = { arController: { canvas: HTMLCanvasElement } | null; init: (ready: () => void) => void; getProjectionMatrix: () => THREE.Matrix4; update: (el: HTMLVideoElement) => void };

export default function MarkerARViewer({ heatmap, onClose }: { heatmap: RotorHeatmap; onClose: () => void }) {
  const mount = useRef<HTMLDivElement>(null);
  const transform = useRef({ diameter: 270, x: 0, z: 0, rotation: 0 });
  const [diameter, setDiameter] = useState('270');
  const [offsetX, setOffsetX] = useState('0');
  const [offsetZ, setOffsetZ] = useState('0');
  const [rotation, setRotation] = useState('0');
  const [status, setStatus] = useState('Starting camera and marker tracker…');
  const [tracked, setTracked] = useState(false);
  const trackedRef = useRef(false);

  useEffect(() => {
    transform.current = {
      diameter: Number(diameter) || 270,
      x: Number(offsetX) || 0,
      z: Number(offsetZ) || 0,
      rotation: Number(rotation) || 0,
    };
  }, [diameter, offsetX, offsetZ, rotation]);

  useEffect(() => {
    let alive = true;
    let frame = 0;
    let contextReady = false;
    let source: ToolkitSource | null = null;
    let context: ToolkitContext | null = null;
    let renderer: THREE.WebGLRenderer | null = null;
    let scene: THREE.Scene | null = null;
    let markerRoot: THREE.Group | null = null;
    let part: THREE.Group | null = null;
    const host = mount.current;
    if (!host) return;

    const fail = (message: string) => { if (alive) setStatus(message); };
    const resize = () => {
      if (!source || !renderer) return;
      source.onResizeElement();
      source.copyElementSizeTo(renderer.domElement);
      if (context?.arController) source.copyElementSizeTo(context.arController.canvas);
      renderer.setSize(window.innerWidth, window.innerHeight, false);
    };

    void (async () => {
      try {
        const tracking = await import('@ar-js-org/ar.js/three.js/build/ar-threex.mjs');
        if (!alive) return;
        tracking.ArToolkitContext.baseURL = '/ar/three.js/';
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: false });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setClearColor(0x000000, 0);
        renderer.domElement.className = 'marker-ar-canvas';
        host.appendChild(renderer.domElement);

        scene = new THREE.Scene();
        const camera = new THREE.Camera();
        scene.add(camera);
        context = new tracking.ArToolkitContext({
          cameraParametersUrl: '/ar/data/data/camera_para.dat', detectionMode: 'mono', canvasWidth: 640, canvasHeight: 480,
        });
        context.init(() => {
          if (!alive || !context) return;
          contextReady = true;
          camera.projectionMatrix.copy(context.getProjectionMatrix());
          setStatus('Camera ready. Center the 40 mm Hiro marker on the rotor hub.');
        });
        markerRoot = new THREE.Group();
        markerRoot.visible = false;
        scene.add(markerRoot);

        const markerWidthM = 0.04;
        new tracking.ArMarkerControls(context, markerRoot, {
          type: 'pattern', patternUrl: '/ar/data/data/patt.hiro', size: markerWidthM, smooth: true, smoothCount: 4,
        });

        part = new THREE.Group();
        markerRoot.add(part);
        const heatLayer = buildHeatmap(heatmap);
        if (!heatLayer) throw new Error('Heatmap grid is empty.');
        part.add(heatLayer);
        const edge = new THREE.Mesh(
          new THREE.RingGeometry(134.3, 135, 128),
          new THREE.MeshBasicMaterial({ color: 0x78e0ce, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }),
        );
        edge.rotation.x = -Math.PI / 2;
        edge.position.y = 22.4;
        part.add(edge);

        source = new tracking.ArToolkitSource({ sourceType: 'webcam', sourceWidth: 640, sourceHeight: 480 });
        source.init(() => {
          if (!alive || !source) return;
          const video = source.domElement;
          if (video) {
            video.style.position = 'fixed';
            video.style.inset = '0';
            video.style.width = '100vw';
            video.style.height = '100vh';
            video.style.objectFit = 'cover';
            video.style.zIndex = '1000';
          }
          resize();
        }, error => fail(`Camera could not start: ${error.message ?? error.name ?? 'permission denied'}. Use HTTPS or localhost and allow camera access.`));
        window.addEventListener('resize', resize);

        const render = () => {
          if (!alive || !renderer || !scene || !context || !source) return;
          frame = requestAnimationFrame(render);
          if (contextReady && source.ready && source.domElement) {
            context.update(source.domElement);
            const isTracked = Boolean(markerRoot?.visible);
            if (trackedRef.current !== isTracked) {
              trackedRef.current = isTracked;
              setTracked(isTracked);
              setStatus(isTracked ? 'Hiro marker tracked · overlay follows its pose.' : 'Show the printed Hiro marker to the camera.');
            }
            if (part) {
              const current = transform.current;
              // Marker sits on the modeled hat top (34 mm); friction face is at 22 mm.
              part.position.set(current.x * 0.001, -0.034, current.z * 0.001);
              part.rotation.y = THREE.MathUtils.degToRad(current.rotation);
              part.scale.setScalar((current.diameter / 270) * 0.001);
            }
            renderer.render(scene, camera);
          }
        };
        render();
      } catch (error) {
        fail(error instanceof Error ? `AR could not initialize: ${error.message}` : 'AR could not initialize.');
      }
    })();

    return () => {
      alive = false;
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', resize);
      source?.dispose();
      renderer?.dispose();
      renderer?.domElement.remove();
      if (scene) scene.traverse(object => {
        const mesh = object as THREE.Mesh;
        mesh.geometry?.dispose();
        const material = mesh.material;
        const disposeMaterial = (item: THREE.Material) => {
          (item as THREE.Material & { map?: THREE.Texture }).map?.dispose();
          item.dispose();
        };
        if (Array.isArray(material)) material.forEach(disposeMaterial);
        else if (material) disposeMaterial(material);
      });
      context = null;
      markerRoot = null;
      part = null;
    };
  }, [heatmap.grid, heatmap.threshold]);

  return <div className="marker-ar" role="dialog" aria-modal="true" aria-label="Tracked AR heatmap">
    <div className="marker-ar-stage" ref={mount} />
    <header className="marker-ar-head"><div><strong>LINEGUARD · TRACKED AR</strong><span className={tracked ? 'tracking-live' : ''}>{tracked ? 'MARKER LOCKED' : 'SEARCHING FOR MARKER'}</span></div><button className="icon-button" aria-label="Close AR" onClick={onClose}><X size={18} /></button></header>
    <aside className="marker-ar-panel">
      <h2>Rotor alignment</h2>
      <p>{status}</p>
      <a href="/marker" target="_blank" rel="noreferrer">Open 40 mm marker print page</a>
      <div className="marker-ar-fields">
        <label>Disc diameter (mm)<input type="number" min="100" max="600" value={diameter} onChange={e => setDiameter(e.target.value)} /></label>
        <label>Center offset X (mm)<input type="number" step="1" value={offsetX} onChange={e => setOffsetX(e.target.value)} /></label>
        <label>Center offset Z (mm)<input type="number" step="1" value={offsetZ} onChange={e => setOffsetZ(e.target.value)} /></label>
        <label>Image rotation (°)<input type="number" min="-180" max="180" value={rotation} onChange={e => setRotation(e.target.value)} /></label>
      </div>
      <p className="marker-ar-caution">The marker pose is tracked live. Put its 40 mm black square at the disc center and set the measured diameter. The heatmap still comes from an unvalidated proxy model and is not a brake-disc defect measurement.</p>
    </aside>
    <span className="marker-ar-crosshair" aria-hidden="true" />
  </div>;
}
