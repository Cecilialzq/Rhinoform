import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

function ramp(t, stops) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [a, ca] = stops[i];
    const [b, cb] = stops[i + 1];
    if (t >= a && t <= b) {
      const f = (t - a) / (b - a || 1);
      return new THREE.Color(ca).lerp(new THREE.Color(cb), f);
    }
  }
  return new THREE.Color(stops[stops.length - 1][1]);
}
const DELTA_RAMP = [
  [0, 0x2b2a26],
  [0.45, 0x9c8a5e],
  [0.75, 0xd7a23a],
  [1, 0xc8502f],
];
const RISK_RAMP = [
  [0, 0x3c6f57],
  [0.5, 0x9c7a23],
  [1, 0xa8402b],
];
const BASE = 0xcdc4b4;

// Full head is displayed; only the nasal ROI (+ collar) deforms.
// source/edited: full-mesh Float32Array. facesFlat: flat index array.
// roiIndices: roi-local -> full vertex (color/scope ROI only); null => color directly.
// delta:{mag(roi-local),max}  risk:{strainPerV(roi-local)}
export default function Studio3D({ source, edited, facesFlat, roiIndices, mode, delta, risk }) {
  const host = useRef(null);
  const ref = useRef({});

  useEffect(() => {
    const el = host.current;
    if (!el || !source) return;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 6000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";
    el.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;

    scene.add(new THREE.HemisphereLight(0xfff3df, 0x111014, 1.7));
    const key = new THREE.DirectionalLight(0xffeccf, 2.8);
    key.position.set(2, 3, 4);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x9fb9ff, 1.2);
    rim.position.set(-3, 1, -3);
    scene.add(rim);
    const fill = new THREE.DirectionalLight(0xffffff, 0.45);
    fill.position.set(0, -2, 2);
    scene.add(fill);

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(source), 3));
    geom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(source.length), 3));
    geom.setIndex(Array.from(facesFlat));
    geom.computeVertexNormals();
    geom.computeBoundingBox();
    const center = geom.boundingBox.getCenter(new THREE.Vector3());
    const size = geom.boundingBox.getSize(new THREE.Vector3()).length();

    const material = new THREE.MeshStandardMaterial({
      color: BASE,
      roughness: 0.55,
      metalness: 0.1,
      side: THREE.DoubleSide,
      vertexColors: false,
      flatShading: false,
    });
    const mesh = new THREE.Mesh(geom, material);
    mesh.position.sub(center);
    scene.add(mesh);

    camera.position.set(size * 0.16, size * 0.04, size * 1.42);
    controls.target.set(0, 0, 0);
    controls.minDistance = size * 0.45;
    controls.maxDistance = size * 4;
    controls.update();

    const resize = () => {
      const w = el.clientWidth,
        h = el.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    controls.autoRotate = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    controls.autoRotateSpeed = 0.42;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      ref.current.raf = requestAnimationFrame(animate);
    };
    ref.current = { renderer, controls, geom, mesh, material, center, ro, raf: 0 };
    animate();

    return () => {
      cancelAnimationFrame(ref.current.raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      el.replaceChildren();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source && source.length, facesFlat]);

  useEffect(() => {
    const s = ref.current;
    if (!s || !s.geom) return;
    const showSource = mode === "before";
    const verts = showSource ? source : edited;
    const pos = s.geom.attributes.position.array;
    pos.set(verts);
    s.geom.attributes.position.needsUpdate = true;
    s.geom.computeVertexNormals();

    const nV = verts.length / 3;
    const col = s.geom.attributes.color.array;
    const base = new THREE.Color(mode === "before" ? 0x9b9485 : BASE);
    let useVC = false;

    const paint = (value, denom, RAMP) => {
      useVC = true;
      for (let v = 0; v < nV; v++) {
        col[v * 3] = base.r;
        col[v * 3 + 1] = base.g;
        col[v * 3 + 2] = base.b;
      }
      const n = roiIndices ? roiIndices.length : nV;
      for (let i = 0; i < n; i++) {
        const fv = roiIndices ? roiIndices[i] : i;
        const c = ramp(value[i] / (denom || 1), RAMP);
        col[fv * 3] = c.r;
        col[fv * 3 + 1] = c.g;
        col[fv * 3 + 2] = c.b;
      }
    };

    if (mode === "delta" && delta) paint(delta.mag, delta.max, DELTA_RAMP);
    else if (mode === "risk" && risk) paint(risk.strainPerV, 0.34, RISK_RAMP);

    if (useVC) {
      s.geom.attributes.color.needsUpdate = true;
      s.material.vertexColors = true;
      s.material.color.set(0xffffff);
    } else {
      s.material.vertexColors = false;
      s.material.color.set(base);
    }
    s.material.needsUpdate = true;
  }, [edited, source, mode, delta, risk, roiIndices]);

  return <div className="studio-viewer" ref={host} aria-label="Full-head nasal preview viewer" />;
}
