'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

export default function ProductionScene({ stopped, manualPause }: { stopped: boolean; manualPause: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const paused = useRef(stopped);
  const critical = useRef(stopped);
  const [error, setError] = useState('');
  useEffect(() => { paused.current = stopped || manualPause; critical.current = stopped; }, [stopped, manualPause]);

  useEffect(() => {
    const container = host.current;
    if (!container) return;
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); }
    catch { setError('3D rendering is unavailable. Inspection alerts remain active.'); return; }
    const compact = window.matchMedia('(max-width: 760px)').matches;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, compact ? 1.25 : 1.5));
    renderer.setClearColor(0x14252e, 1);
    container.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-7.2, 7.2, 3, -3, .1, 100);
    camera.position.set(0, 9, 13); camera.lookAt(0, 0, 0);
    scene.add(new THREE.HemisphereLight(0xc9eaff, 0x353224, 3));
    const light = new THREE.DirectionalLight(0xffffff, 3); light.position.set(3, 8, 5); scene.add(light);
    const material = (color: number) => new THREE.MeshStandardMaterial({ color, metalness: .55, roughness: .35 });
    const steel = material(0x97aeb8), teal = material(0x238e85), dark = material(0x334e5d);
    const belt = new THREE.Mesh(new THREE.BoxGeometry(12, .25, 2.1), dark); belt.position.y = -.35; scene.add(belt);
    for (let x = -5.7; x <= 5.7; x += .45) {
      const roller = new THREE.Mesh(new THREE.CylinderGeometry(.13, .13, 2.15, 12), steel);
      roller.rotation.x = Math.PI / 2; roller.position.set(x, -.12, 0); scene.add(roller);
    }
    for (const x of [-4.8, -2.4, 0, 2.4, 4.8]) {
      const base = new THREE.Mesh(new THREE.BoxGeometry(1.65, .4, 1), teal); base.position.set(x, -.5, -1.65); scene.add(base);
      const machine = new THREE.Mesh(new THREE.BoxGeometry(1.4, 1.25, .7), dark); machine.position.set(x, .3, -1.65); scene.add(machine);
    }
    const furnace = new THREE.Mesh(new THREE.CylinderGeometry(.5, .5, .8, 24), new THREE.MeshStandardMaterial({ color: 0xd98736, emissive: 0x73370f, metalness: .35, roughness: .45 }));
    furnace.position.set(-4.8, .75, -1.65); scene.add(furnace);
    const press = new THREE.Mesh(new THREE.BoxGeometry(.45, 1.6, .45), steel); press.position.set(-2.4, .7, -1); scene.add(press);
    for (const z of [-1.1, 1.1]) { const pole = new THREE.Mesh(new THREE.BoxGeometry(.15, 1.8, .15), teal); pole.position.set(0, .75, z); scene.add(pole); }
    const arch = new THREE.Mesh(new THREE.BoxGeometry(.2, .15, 2.35), teal); arch.position.set(0, 1.7, 0); scene.add(arch);
    const scannerMaterial = new THREE.MeshStandardMaterial({ color: 0x49dac5, emissive: 0x14735e, transparent: true, opacity: .6 });
    const scan = new THREE.Mesh(new THREE.BoxGeometry(.15, 1.6, 2.1), scannerMaterial); scan.position.set(0, .7, 0); scene.add(scan);
    const parts: THREE.Group[] = [];
    const inspectionPaint = steel.clone();
    for (let i = 0; i < 6; i++) {
      const part = new THREE.Group();
      part.add(new THREE.Mesh(new THREE.CylinderGeometry(.48, .48, .16, 40), i === 2 ? inspectionPaint : steel));
      const hub = new THREE.Mesh(new THREE.CylinderGeometry(.2, .2, .2, 28), dark); hub.position.y = .12; part.add(hub);
      part.position.set(-5.4 + i * 2, .14, 0); scene.add(part); parts.push(part);
    }
    let dirty = true;
    const resize = () => { const width = Math.max(container.clientWidth, 1); const height = width < 600 ? 220 : 280; renderer.setSize(width, height); camera.top = 7.2 * height / width; camera.bottom = -camera.top; camera.updateProjectionMatrix(); dirty = true; };
    const observer = new ResizeObserver(resize); observer.observe(container); resize();
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    let visible = true;
    const visibility = new IntersectionObserver(entries => { visible = entries[0]?.isIntersecting ?? true; if (visible) dirty = true; });
    visibility.observe(container);
    let frame = 0, previous = 0, wasCritical = false, wasPaused = false;
    const tick = (now: number) => {
      if (now - previous < (compact ? 41 : 33)) { frame = requestAnimationFrame(tick); return; }
      const dt = Math.min((now - previous) / 1000, .06); previous = now;
      if (!document.hidden && visible) {
        dirty ||= critical.current !== wasCritical || paused.current !== wasPaused;
        if (critical.current && !wasCritical) parts[2].position.x = 0;
        wasCritical = critical.current;
        wasPaused = paused.current;
        inspectionPaint.color.setHex(critical.current ? 0xee4c3d : 0x97aeb8);
        if (!paused.current && !reduced.matches) for (const part of parts) {
          part.position.x += dt * .7;
          if (part.position.x > 5.8) part.position.x = -5.8;
        }
        scannerMaterial.color.setHex(paused.current ? 0xff584d : 0x49dac5);
        scannerMaterial.emissive.setHex(paused.current ? 0x962019 : 0x14735e);
        if (dirty || (!paused.current && !reduced.matches)) renderer.render(scene, camera);
        dirty = false;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame); observer.disconnect(); visibility.disconnect();
      const geometries = new Set<THREE.BufferGeometry>(); const materials = new Set<THREE.Material>();
      scene.traverse(object => { if (object instanceof THREE.Mesh) { geometries.add(object.geometry); for (const m of Array.isArray(object.material) ? object.material : [object.material]) materials.add(m); } });
      geometries.forEach(g => g.dispose()); materials.forEach(m => m.dispose());
      renderer.dispose(); renderer.domElement.remove();
    };
  }, []);

  return <><div ref={host} className="production-scene" role="img" aria-label={`Manufacturing conveyor. ${stopped ? 'Stopped for a critical defect.' : 'Production simulation.'}`} />{error && <p role="status">{error}</p>}</>;
}
