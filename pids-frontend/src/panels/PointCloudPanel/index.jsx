import { useRef, useEffect, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import styles from './PointCloudPanel.module.css'

const VERT = /* glsl */`
  uniform float uMode;     // 0=intensity 1=height 2=cluster
  attribute float intensity;
  attribute float clusterId;
  varying vec3 vColor;

  vec3 ramp(float u) {
    vec3 c0 = vec3(0.000, 0.267, 1.000);
    vec3 c1 = vec3(0.000, 0.667, 1.000);
    vec3 c2 = vec3(0.000, 1.000, 0.533);
    vec3 c3 = vec3(1.000, 0.800, 0.000);
    vec3 c4 = vec3(1.000, 0.133, 0.000);
    u = clamp(u, 0.0, 1.0);
    if (u < 0.25) return mix(c0, c1, u / 0.25);
    if (u < 0.50) return mix(c1, c2, (u - 0.25) / 0.25);
    if (u < 0.75) return mix(c2, c3, (u - 0.50) / 0.25);
                  return mix(c3, c4, (u - 0.75) / 0.25);
  }

  vec3 clusterColor(float id) {
    float h = mod(id * 0.618033, 1.0);
    float s = 0.8; float v = 0.95;
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + K.xyz) * 6.0 - K.www);
    return v * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
  }

  void main() {
    if (uMode < 0.5)       vColor = ramp(intensity);
    else if (uMode < 1.5)  vColor = ramp((position.y + 2.0) / 4.0);
    else                   vColor = clusterColor(clusterId);

    gl_Position  = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = 2.0;
  }
`

const FRAG = /* glsl */`
  varying vec3 vColor;
  void main() {
    float d = length(gl_PointCoord - vec2(0.5));
    if (d > 0.5) discard;
    gl_FragColor = vec4(vColor, 0.9);
  }
`

const COLOR_MODES = ['intensity', 'height', 'cluster']

export default function PointCloudPanel({ lidarData, onClose }) {
  const { frameRef, meta, clusters } = lidarData
  const mountRef   = useRef(null)
  const sceneRef   = useRef({})
  const [colorMode, setColorMode] = useState(0)

  useEffect(() => {
    const el  = mountRef.current
    const w   = el.offsetWidth
    const h   = el.offsetHeight

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(w, h)
    renderer.setClearColor(0x080c10)
    renderer.autoClear = true
    el.appendChild(renderer.domElement)

    const scene  = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 500)
    camera.position.set(0, 10, 25)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05

    scene.add(new THREE.GridHelper(60, 60, 0x1e3a5f, 0x0d1a2e))

    const MAX  = 131072
    const pos  = new Float32Array(MAX * 3)
    const intn = new Float32Array(MAX)
    const cids = new Float32Array(MAX)

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position',  new THREE.BufferAttribute(pos,  3))
    geo.setAttribute('intensity', new THREE.BufferAttribute(intn, 1))
    geo.setAttribute('clusterId', new THREE.BufferAttribute(cids, 1))
    geo.setDrawRange(0, 0)

    const mat = new THREE.ShaderMaterial({
      uniforms:       { uMode: { value: 0 } },
      vertexShader:   VERT,
      fragmentShader: FRAG,
      transparent:    true,
    })

    const cloud = new THREE.Points(geo, mat)
    scene.add(cloud)
    sceneRef.current = { renderer, scene, camera, controls, geo, mat, cloud }

    const onResize = () => {
      const nw = el.offsetWidth
      const nh = el.offsetHeight
      camera.aspect = nw / nh
      camera.updateProjectionMatrix()
      renderer.setSize(nw, nh)
    }
    const ro = new ResizeObserver(onResize)
    ro.observe(el)

    let raf
    const animate = () => {
      raf = requestAnimationFrame(animate)
      controls.update()

      // Sync point data
      const frame = frameRef.current
      if (frame && frame.n > 0) {
        const { positions: src, intensities: isrc, n } = frame
        pos.set(src.subarray(0, n * 3))
        intn.set(isrc.subarray(0, n))
        geo.attributes.position.needsUpdate  = true
        geo.attributes.intensity.needsUpdate = true
        geo.setDrawRange(0, n)
      }

      renderer.render(scene, camera)
    }
    animate()

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      renderer.dispose()
      el.removeChild(renderer.domElement)
    }
  }, [frameRef])

  // Sync color mode uniform
  useEffect(() => {
    if (sceneRef.current.mat) {
      sceneRef.current.mat.uniforms.uMode.value = colorMode
    }
  }, [colorMode])

  return (
    <div className={styles.overlay}>
      <div className={styles.toolbar}>
        <span className={styles.title}>◈ 3D point cloud</span>
        <div className={styles.modes}>
          {COLOR_MODES.map((m, i) => (
            <button
              key={m}
              className={`btn ${colorMode === i ? 'active' : ''}`}
              onClick={() => setColorMode(i)}
            >
              {m}
            </button>
          ))}
        </div>
        <div className={styles.info}>
          <span>{meta.n.toLocaleString()} pts</span>
          <span>{clusters.length} clusters</span>
        </div>
        <button className="btn" onClick={onClose}>✕ Close</button>
      </div>
      <div ref={mountRef} className={styles.viewport} />
    </div>
  )
}
