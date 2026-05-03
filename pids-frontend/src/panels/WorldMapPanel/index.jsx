import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import PanelShell from '@/components/PanelShell'
import styles from './WorldMapPanel.module.css'

const MAX_POINTS = 131072
const MAX_CELLS = 2500
const MODES = [
  { key: 0, label: 'HEAT', title: 'thermal heat proxy' },
  { key: 1, label: 'INT', title: 'lidar intensity' },
  { key: 2, label: 'HT', title: 'height' },
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
    if (uMode < 1.5) vColor = ramp(intensity);
    else             vColor = ramp((position.y + 2.0) / 4.0);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = 2.2;
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

export default function WorldMapPanel({ lidarData, fusionData, thermalData, onPopOut3D }) {
  const { frameRef, meta: lidarMeta, connState: lidarState } = lidarData
  const { frame: fusionFrame } = fusionData
  const { meta: thermalMeta } = thermalData
  const mountRef = useRef(null)
  const sceneRef = useRef(null)
  const [mode, setMode] = useState(0)

  const tracks = useMemo(() => normalizeTracks(fusionFrame.tracks ?? []), [fusionFrame.tracks])
  const subtitle = `${lidarMeta.n.toLocaleString()} PTS`

  useEffect(() => {
    const el = mountRef.current
    if (!el) return

    const width = Math.max(1, el.offsetWidth)
    const height = Math.max(1, el.offsetHeight)
    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5))
    renderer.setSize(width, height)
    renderer.setClearColor(0x080c10)
    el.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.fog = new THREE.Fog(0x080c10, 55, 120)

    const camera = new THREE.PerspectiveCamera(58, width / height, 0.1, 220)
    camera.position.set(-18, 12, 24)
    camera.lookAt(0, 0, 8)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = false
    controls.target.set(0, 0, 8)
    controls.minDistance = 5
    controls.maxDistance = 90
    controls.addEventListener('change', renderScene)

    scene.add(new THREE.GridHelper(70, 35, 0x1e3a5f, 0x102033))
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
      uniforms: { uMode: { value: mode } },
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
      depthWrite: false,
    })
    const cloud = new THREE.Points(geo, mat)
    scene.add(cloud)

    const cellGroup = new THREE.Group()
    const trackGroup = new THREE.Group()
    scene.add(cellGroup)
    scene.add(trackGroup)

    function renderScene() {
      renderer.render(scene, camera)
    }

    const ro = new ResizeObserver(() => {
      const nextW = Math.max(1, el.offsetWidth)
      const nextH = Math.max(1, el.offsetHeight)
      camera.aspect = nextW / nextH
      camera.updateProjectionMatrix()
      renderer.setSize(nextW, nextH)
      renderScene()
    })
    ro.observe(el)

    sceneRef.current = { renderer, controls, geo, mat, pos, intn, cellGroup, trackGroup, renderScene }
    renderScene()

    return () => {
      ro.disconnect()
      controls.dispose()
      disposeGroup(cellGroup)
      disposeGroup(trackGroup)
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
  }, [frameRef, lidarMeta.seq])

  useEffect(() => {
    const ctx = sceneRef.current
    if (!ctx) return
    ctx.mat.uniforms.uMode.value = mode
    ctx.renderScene()
  }, [mode])

  useEffect(() => {
    const ctx = sceneRef.current
    if (!ctx) return
    drawFusionCells(ctx.cellGroup, fusionFrame.map)
    drawTracks(ctx.trackGroup, tracks)
    ctx.renderScene()
  }, [fusionFrame.map, tracks])

  const controls = (
    <div className={styles.controls}>
      {MODES.map(item => (
        <button
          key={item.key}
          className={`btn ${mode === item.key ? 'active' : ''}`}
          onClick={() => setMode(item.key)}
          title={item.title}
        >
          {item.label}
        </button>
      ))}
      {onPopOut3D && <button className="btn" onClick={onPopOut3D} title="Open full point cloud">POP</button>}
    </div>
  )

  return (
    <PanelShell title="WORLD 3D" subtitle={subtitle} modality="lidar" connState={lidarState} controls={controls}>
      <div className={styles.viewport}>
        <div ref={mountRef} className={styles.scene} />
        <div className={styles.readout}>
          <span className={styles.live}>{lidarMeta.n.toLocaleString()} pts</span>
          <span className={styles.heat}>{thermalMeta.tMin.toFixed(1)}-{thermalMeta.tMax.toFixed(1)} C</span>
          <span>{tracks.length} trk</span>
        </div>
      </div>
    </PanelShell>
  )
}

function drawFusionCells(group, map) {
  disposeGroup(group)
  if (!map) return
  const width = map.width ?? map.w ?? 0
  const height = map.height ?? map.h ?? 0
  const cells = Array.isArray(map.cells) ? map.cells : null
  if (!width || !height || !cells?.length) return

  const res = map.resolution ?? map.res ?? 0.5
  const origin = map.origin ?? {}
  const ox = origin.x ?? map.origin_x ?? -(width * res) / 2
  const oz = origin.z ?? origin.y ?? map.origin_z ?? -(height * res) / 2
  const stride = Math.max(1, Math.ceil(Math.sqrt(cells.length / MAX_CELLS)))

  for (let row = 0; row < height; row += stride) {
    for (let col = 0; col < width; col += stride) {
      const cell = cells[row * width + col]
      const occ = cellValue(cell)
      if (occ <= 0.08) continue
      const temp = cellTemp(cell)
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(res * stride, Math.max(0.03, occ * 0.45), res * stride),
        new THREE.MeshBasicMaterial({ color: cellColor(occ, temp), transparent: true, opacity: 0.52 })
      )
      mesh.position.set(ox + col * res, occ * 0.24, oz + row * res)
      group.add(mesh)
    }
  }
}

function drawTracks(group, tracks) {
  disposeGroup(group)
  tracks.forEach(track => {
    const color = confidenceColor(track.confidence)
    const mat = new THREE.MeshBasicMaterial({ color })
    const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 2.2, 12), mat)
    pin.position.set(track.x, 1.1, track.z)
    group.add(pin)
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.8, 0.025, 8, 32),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 })
    )
    ring.rotation.x = Math.PI / 2
    ring.position.set(track.x, 0.05, track.z)
    group.add(ring)
  })
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

  group.add(new THREE.ArrowHelper(
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(0, 0.76, 0),
    2.4,
    0x00d4ff,
    0.28,
    0.14
  ))
  return group
}

function makeRangeRings() {
  const group = new THREE.Group()
  const material = new THREE.LineBasicMaterial({ color: 0x1e3a5f, transparent: true, opacity: 0.5 })
  ;[10, 20, 30].forEach(radius => {
    const pts = []
    for (let i = 0; i <= 96; i++) {
      const a = (i / 96) * Math.PI * 2
      pts.push(new THREE.Vector3(Math.cos(a) * radius, 0.025, Math.sin(a) * radius))
    }
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), material))
  })
  return group
}

function normalizeTracks(items) {
  return items.map((item, i) => {
    const pos = item.position ?? item.centroid ?? item
    const x = Number(pos.x ?? Math.sin((item.bearing ?? 0) * Math.PI / 180) * (item.range ?? 0))
    const z = Number(pos.z ?? pos.y ?? Math.cos((item.bearing ?? 0) * Math.PI / 180) * (item.range ?? 0))
    return { id: item.id ?? item.track_id ?? i + 1, x, z, confidence: clamp01(item.confidence ?? item.score ?? 0.45) }
  })
}

function disposeGroup(group) {
  while (group.children.length) {
    const child = group.children.pop()
    child.geometry?.dispose?.()
    child.material?.dispose?.()
  }
}

function cellValue(cell) {
  if (typeof cell === 'number') return cell
  return cell?.occupancy ?? cell?.occ ?? cell?.value ?? 0
}

function cellTemp(cell) {
  if (typeof cell === 'number') return null
  return cell?.temperature ?? cell?.temp ?? null
}

function cellColor(occ, temp) {
  if (temp != null) {
    const hot = clamp01((temp - 15) / 45)
    return new THREE.Color(0.25 + hot * 0.75, 0.85 - hot * 0.55, 0.12)
  }
  return new THREE.Color(0.0, 0.45 + occ * 0.45, 1.0)
}

function confidenceColor(confidence) {
  if (confidence >= 0.8) return 0xff2244
  if (confidence >= 0.6) return 0xff8c42
  if (confidence >= 0.4) return 0xffcc00
  return 0x00e676
}

function clamp01(v) {
  return Math.max(0, Math.min(1, Number(v) || 0))
}
