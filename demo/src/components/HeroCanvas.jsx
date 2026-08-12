import React, { useEffect, useRef } from "react";
import * as THREE from "three";

// Luminous point-cloud + wireframe of the real nasal ROI, slowly rotating.
// This is the native WebGL substitute for the requested cinematic hero video:
// it renders the actual frozen-case geometry rather than stock footage.
export default function HeroCanvas() {
  const host = useRef(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 4000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";
    el.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    let raf = 0;
    let disposed = false;

    fetch("/bundle/ridge_browser.json", { cache: "no-store" })
      .then((r) => r.json())
      .then((bundle) => {
        if (disposed) return;
        const verts = new Float32Array(bundle.source.flat());
        const faces = bundle.faces.flat();

        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(verts, 3));
        geom.setIndex(faces);
        geom.computeVertexNormals();
        geom.computeBoundingBox();
        const center = geom.boundingBox.getCenter(new THREE.Vector3());
        const size = geom.boundingBox.getSize(new THREE.Vector3()).length();

        const points = new THREE.Points(
          geom,
          new THREE.PointsMaterial({
            color: 0xf3e9d6,
            size: size * 0.006,
            transparent: true,
            opacity: 0.85,
            sizeAttenuation: true,
          })
        );
        const wire = new THREE.LineSegments(
          new THREE.WireframeGeometry(geom),
          new THREE.LineBasicMaterial({ color: 0xc89a5a, transparent: true, opacity: 0.16 })
        );
        [points, wire].forEach((o) => {
          o.position.sub(center);
          group.add(o);
        });
        group.rotation.x = -0.1;

        camera.position.set(0, size * 0.05, size * 1.7);
        camera.lookAt(0, 0, 0);
        resize();
      })
      .catch(() => {});

    function resize() {
      const w = el.clientWidth,
        h = el.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    const animate = () => {
      group.rotation.y += reduce ? 0 : 0.0024;
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.dispose();
      el.replaceChildren();
    };
  }, []);

  return <div className="hero__canvas" ref={host} aria-hidden="true" />;
}
