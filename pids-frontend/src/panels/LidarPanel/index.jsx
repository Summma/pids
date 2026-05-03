import { useRef, useEffect, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import PanelShell from '@/components/PanelShell'
import styles from './LidarPanel.module.css'

const MAX_POINTS = 131072
const COLOR_MODES = [
  { key: 0, label: 'Intensity', title: 'intensity' },
  { key: 1, label: 'Height', title: 'height' },
]

const VERT = /* glsl */`
  uniform float uMode;
  attribute float intensity;
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

  void main() {
    if (uMode < 0.5) vColor = ramp(intensity);
    else             vColor = ramp((position.y + 2.0) / 4.0);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = 2.0;
  }
`

const FRAG = /* glsl */`
  varying vec3 vColor;
  void main() {
    float d = length(gl_PointCoord - vec2(0.5));
    if (d > 0.5) discard;
    gl_FragColor = vec4(vColor, 0.92);
  }
`

export default function LidarPanel({ lidarData }) {
  const { connState, frameRef, meta } = lidarData
  const mountRef = useRef(null)
  const sceneRef = useRef(null)
  const [colorMode, setColorMode] = useState(0)

  useEffect(() => {
    const el = mountRef.current
    if (!el) return

    const width = Math.max(1, el.offsetWidth)
    const height = Math.max(1, el.offsetHeight)
    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setSize(width, height)
    renderer.setClearColor(0x080c10)
    el.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.fog = new THREE.Fog(0x080c10, 45, 100)

    const camera = new THREE.PerspectiveCamera(58, width / height, 0.1, 180)
    camera.position.set(-14, 8, 18)
    camera.lookAt(0, 0, 0)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = false
    controls.target.set(0, 0, 8)
    controls.minDistance = 4
    controls.maxDistance = 80
    controls.addEventListener('change', () => renderScene())

    const grid = new THREE.GridHelper(60, 30, 0x1e3a5f, 0x102033)
    scene.add(grid)
    scene.add(makeSensorModel())
    scene.add(makeRangeRings())
    scene.add(new THREE.AmbientLight(0x88aacc, 0.9))

    const pos = new Float32Array(MAX_POINTS * 3)
    const intn = new Float32Array(MAX_POINTS)
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    geo.setAttribute('intensity', new THREE.BufferAttribute(intn, 1))
    geo.setDrawRange(0, 0)

    const mat = new THREE.ShaderMaterial({
      uniforms: { uMode: { value: colorMode } },
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
      depthWrite: false,
    })

    const cloud = new THREE.Points(geo, mat)
    scene.add(cloud)

    function renderScene() {
      renderer.render(scene, camera)
    }

    const resizeObserver = new ResizeObserver(() => {
      const nextW = Math.max(1, el.offsetWidth)
      const nextH = Math.max(1, el.offsetHeight)
      camera.aspect = nextW / nextH
      camera.updateProjectionMatrix()
      renderer.setSize(nextW, nextH)
      renderScene()
    })
    resizeObserver.observe(el)

    sceneRef.current = { renderer, scene, camera, controls, geo, mat, pos, intn, renderScene }
    renderScene()

    return () => {
      resizeObserver.disconnect()
      controls.dispose()
      geo.dispose()
      mat.dispose()
      renderer.dispose()
      el.removeChild(renderer.domElement)
      sceneRef.current = null
    }
  }, [])

  useEffect(() => {
    const ctx = sceneRef.current
    const frame = frameRef.current
    if (!ctx || !frame || frame.n === 0) return

    const n = Math.min(frame.n, MAX_POINTS)
    ctx.pos.set(frame.positions.subarray(0, n * 3))
    ctx.intn.set(frame.intensities.subarray(0, n))
    ctx.geo.attributes.position.needsUpdate = true
    ctx.geo.attributes.intensity.needsUpdate = true
    ctx.geo.setDrawRange(0, n)
    ctx.renderScene()
  }, [frameRef, meta.seq])

  useEffect(() => {
    const ctx = sceneRef.current
    if (!ctx) return
    ctx.mat.uniforms.uMode.value = colorMode
    ctx.renderScene()
  }, [colorMode])

  const controls = (
    <div className={styles.controls}>
      {COLOR_MODES.map(mode => (
        <button
          key={mode.key}
          className={`btn ${colorMode === mode.key ? 'active' : ''}`}
          onClick={() => setColorMode(mode.key)}
          title={mode.title}
        >
          {mode.label}
        </button>
      ))}
    </div>
  )

  return (
    <PanelShell title="Lidar" subtitle="3D sensor view" modality="lidar" connState={connState} controls={controls}>
      <div className={styles.viewport}>
        <div ref={mountRef} className={styles.scene} />
        <div className={styles.ptCount}>{meta.n.toLocaleString()} pts</div>
      </div>
    </PanelShell>
  )
}

function makeSensorModel() {
  const group = new THREE.Group()

  const base = new THREE.Mesh(
    new THREE.CylinderGeometry(0.55, 0.7, 0.35, 32),
    new THREE.MeshBasicMaterial({ color: 0x1b3147 })
  )
  base.position.y = 0.18
  group.add(base)

  const head = new THREE.Mesh(
    new THREE.CylinderGeometry(0.34, 0.34, 0.62, 32),
    new THREE.MeshBasicMaterial({ color: 0x00d4ff, wireframe: true })
  )
  head.position.y = 0.72
  group.add(head)

  const mast = new THREE.Mesh(
    new THREE.CylinderGeometry(0.08, 0.08, 0.8, 16),
    new THREE.MeshBasicMaterial({ color: 0x6a8fa8 })
  )
  mast.position.y = 0.45
  group.add(mast)

  const forward = new THREE.ArrowHelper(
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(0, 0.75, 0),
    2.2,
    0x00d4ff,
    0.28,
    0.14
  )
  group.add(forward)

  return group
}

function makeRangeRings() {
  const group = new THREE.Group()
  const material = new THREE.LineBasicMaterial({ color: 0x1e3a5f, transparent: true, opacity: 0.55 })
  ;[10, 20, 30].forEach(radius => {
    const points = []
    for (let i = 0; i <= 96; i++) {
      const a = (i / 96) * Math.PI * 2
      points.push(new THREE.Vector3(Math.cos(a) * radius, 0.02, Math.sin(a) * radius))
    }
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material))
  })
  return group
}
