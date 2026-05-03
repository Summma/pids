import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js'
import { LineMaterial } from 'three/addons/lines/LineMaterial.js'
import PanelShell from '@/components/PanelShell'
import styles from './WorldMapPanel.module.css'

const MAX_POINTS = 131072
const VIEW_DISTANCE = 15
const VIEW_ELEVATION_DEG = 34
const VIEW_AZIMUTH_DEG = 42
const FOV_DEPTH_M = 10
const GUI_DEFAULT_THERMAL_CALIBRATION = {
  intr: { width: 640, height: 512, fx: 686, fy: 686, cx: 320, cy: 256 },
  extr: { tx: 0, ty: 0, tz: 0, rollDeg: 0, pitchDeg: 0, yawDeg: 0 },
  source: 'gui_default',
}
const MODES = [
  { key: 0, label: 'Thermal', title: 'thermal camera projection' },
  { key: 1, label: 'Range', title: 'distance from lidar origin' },
  { key: 2, label: 'Height', title: 'height above sensor ground plane' },
  { key: 3, label: 'Intensity', title: 'lidar intensity / reflectivity proxy' },
]

const VERT = /* glsl */`
  uniform float uMode;
  uniform float uHasThermal;
  uniform float uRangeLo;
  uniform float uRangeHi;
  uniform float uHeightLo;
  uniform float uHeightHi;
  uniform float uIntensityLo;
  uniform float uIntensityHi;
  uniform float uPointSize;
  attribute float intensity;
  attribute float thermalValue;
  attribute float thermalValid;
  varying vec3 vColor;

  vec3 turboLike(float u) {
    u = clamp(u, 0.0, 1.0);
    return vec3(
      clamp(1.5 - abs(4.0 * u - 3.0), 0.0, 1.0),
      clamp(1.5 - abs(4.0 * u - 2.0), 0.0, 1.0),
      clamp(1.5 - abs(4.0 * u - 1.0), 0.0, 1.0)
    );
  }

  float rescale(float value, float lo, float hi) {
    return clamp((value - lo) / max(hi - lo, 0.0001), 0.0, 1.0);
  }

  void main() {
    float rangeM = length(position.xyz);
    if (uMode < 0.5) {
      if (uHasThermal < 0.5)      vColor = turboLike(rescale(rangeM, uRangeLo, uRangeHi));
      else if (thermalValid > 0.5) vColor = turboLike(thermalValue);
      else                         vColor = vec3(0.18, 0.18, 0.20);
    } else if (uMode < 1.5) {
      vColor = turboLike(rescale(rangeM, uRangeLo, uRangeHi));
    } else if (uMode < 2.5) {
      vColor = turboLike(rescale(position.y, uHeightLo, uHeightHi));
    } else {
      vColor = turboLike(rescale(intensity, uIntensityLo, uIntensityHi));
    }
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uPointSize;
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

const WorldMapPanel = forwardRef(function WorldMapPanel({
  lidarData,
  thermalData,
  selectedObjectKey = '',
  thermalCalibrationOverride = null,
}, ref) {
  const {
    frameRef,
    meta: lidarMeta,
    connState: lidarState,
    detections = [],
  } = lidarData
  const { frameRef: thermalFrameRef, meta: thermalMeta, error: thermalError, connState: thermalState } = thermalData
  const mountRef = useRef(null)
  const sceneRef = useRef(null)
  const [mode, setMode] = useState(0)

  const boxes = useMemo(() => normalizeDetections(detections), [detections])
  const thermalCalibration = useMemo(
    () => normalizeThermalCalibration(thermalCalibrationOverride ?? thermalMeta.calibration),
    [thermalCalibrationOverride, thermalMeta.calibration],
  )
  const thermalConfirmed = boxes.filter(box => box.thermalScore >= 0.62 && box.thermalCoverage >= 0.12).length
  const thermalStatus = thermalError || (thermalState === 'live' ? `${thermalMeta.tMin.toFixed(1)}-${thermalMeta.tMax.toFixed(1)} C` : 'thermal offline')
  const subtitle = `${lidarMeta.n.toLocaleString()} pts · ${boxes.length} ${boxes.length === 1 ? 'box' : 'boxes'} · ${thermalConfirmed} heat-supported`

  useEffect(() => {
    const el = mountRef.current
    if (!el) return

    const width = Math.max(1, el.offsetWidth)
    const height = Math.max(1, el.offsetHeight)
    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: true,
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setSize(width, height)
    renderer.setClearColor(0x0a0a0c)
    el.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0a0a0c)

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 500)
    const defaultCamera = alejandroCameraPosition()
    camera.position.set(defaultCamera.x, defaultCamera.y, defaultCamera.z)
    camera.lookAt(0, 0, 0)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = false
    controls.target.set(0, 0, 0)
    controls.minDistance = 2
    controls.maxDistance = 120
    controls.addEventListener('change', renderScene)

    scene.add(makeAlejandroGrid())
    scene.add(makeLidarAxes())
    const fovGroup = makeThermalFovGroup(FOV_DEPTH_M, GUI_DEFAULT_THERMAL_CALIBRATION)
    scene.add(fovGroup)
    scene.add(new THREE.AmbientLight(0xffffff, 0.8))

    const pos = new Float32Array(MAX_POINTS * 3)
    const intn = new Float32Array(MAX_POINTS)
    const thermalValue = new Float32Array(MAX_POINTS)
    const thermalValid = new Float32Array(MAX_POINTS)
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    geo.setAttribute('intensity', new THREE.BufferAttribute(intn, 1))
    geo.setAttribute('thermalValue', new THREE.BufferAttribute(thermalValue, 1))
    geo.setAttribute('thermalValid', new THREE.BufferAttribute(thermalValid, 1))
    geo.setDrawRange(0, 0)

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uMode: { value: mode },
        uHasThermal: { value: 0 },
        uRangeLo: { value: 0 },
        uRangeHi: { value: 30 },
        uHeightLo: { value: -1 },
        uHeightHi: { value: 3 },
        uIntensityLo: { value: 0 },
        uIntensityHi: { value: 1 },
        uPointSize: { value: 2 },
      },
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
      depthWrite: false,
    })
    const cloud = new THREE.Points(geo, mat)
    scene.add(cloud)

    const boxGroup = new THREE.Group()
    scene.add(boxGroup)

    function renderScene() {
      renderer.render(scene, camera)
    }

    const ro = new ResizeObserver(() => {
      const nextW = Math.max(1, el.offsetWidth)
      const nextH = Math.max(1, el.offsetHeight)
      camera.aspect = nextW / nextH
      camera.updateProjectionMatrix()
      renderer.setSize(nextW, nextH)
      updateBoxMaterialResolution(boxGroup, nextW, nextH)
      renderScene()
    })
    ro.observe(el)

    sceneRef.current = {
      scene,
      renderer,
      controls,
      geo,
      mat,
      pos,
      intn,
      thermalValue,
      thermalValid,
      boxGroup,
      fovGroup,
      renderScene,
    }
    renderScene()

    return () => {
      ro.disconnect()
      controls.dispose()
      disposeGroup(boxGroup)
      disposeGroup(sceneRef.current?.fovGroup ?? fovGroup)
      geo.dispose()
      mat.dispose()
      renderer.dispose()
      el.removeChild(renderer.domElement)
      sceneRef.current = null
    }
  }, [])

  useEffect(() => {
    const ctx = sceneRef.current
    if (!ctx) return
    if (ctx.fovGroup) {
      ctx.scene.remove(ctx.fovGroup)
      disposeGroup(ctx.fovGroup)
    }
    ctx.fovGroup = makeThermalFovGroup(FOV_DEPTH_M, thermalCalibration)
    ctx.scene.add(ctx.fovGroup)
    ctx.renderScene()
  }, [thermalCalibration])

  useEffect(() => {
    const ctx = sceneRef.current
    const frame = frameRef.current
    if (!ctx || !frame || frame.n === 0) return

    const n = Math.min(frame.n, MAX_POINTS)
    for (let i = 0; i < n; i++) {
      const src = i * 3
      const dst = i * 3
      const xForward = frame.positions[src]
      const yLeft = frame.positions[src + 1]
      const zUp = frame.positions[src + 2]
      const scenePoint = lidarToScene([xForward, yLeft, zUp])
      ctx.pos[dst] = scenePoint[0]
      ctx.pos[dst + 1] = scenePoint[1]
      ctx.pos[dst + 2] = scenePoint[2]
    }
    ctx.intn.set(frame.intensities.subarray(0, n))
    updateScalarBounds(ctx, frame, n)
    updateThermalProjection(ctx, frame, thermalFrameRef.current, n, thermalCalibration)
    ctx.geo.attributes.position.needsUpdate = true
    ctx.geo.attributes.intensity.needsUpdate = true
    ctx.geo.attributes.thermalValue.needsUpdate = true
    ctx.geo.attributes.thermalValid.needsUpdate = true
    ctx.geo.setDrawRange(0, n)
    ctx.renderScene()
  }, [frameRef, thermalFrameRef, lidarMeta.seq, thermalCalibration])

  useEffect(() => {
    const ctx = sceneRef.current
    const frame = frameRef.current
    if (!ctx || !frame || frame.n === 0) return
    const n = Math.min(frame.n, MAX_POINTS)
    updateThermalProjection(ctx, frame, thermalFrameRef.current, n, thermalCalibration)
    ctx.geo.attributes.thermalValue.needsUpdate = true
    ctx.geo.attributes.thermalValid.needsUpdate = true
    ctx.renderScene()
  }, [frameRef, thermalFrameRef, thermalMeta.seq, thermalCalibration])

  useEffect(() => {
    const ctx = sceneRef.current
    if (!ctx) return
    ctx.mat.uniforms.uMode.value = mode
    ctx.renderScene()
  }, [mode])

  useEffect(() => {
    const ctx = sceneRef.current
    if (!ctx) return
    drawDetectionBoxes(ctx, boxes, selectedObjectKey)
    ctx.renderScene()
  }, [boxes, selectedObjectKey])

  useImperativeHandle(ref, () => ({
    capture() {
      const ctx = sceneRef.current
      if (!ctx) return null
      ctx.renderScene()
      return ctx.renderer.domElement.toDataURL('image/jpeg', 0.86)
    },
  }), [lidarMeta.seq, thermalMeta.seq, boxes, mode, selectedObjectKey])

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
    </div>
  )

  return (
    <PanelShell title="Lidar and thermal" subtitle={subtitle} modality="lidar" connState={lidarState} bare>
      <div className={styles.viewport}>
        <div className={styles.controlsOverlay}>
          {controls}
        </div>
        <div ref={mountRef} className={styles.scene} />
        <div className={styles.readout}>
          <span className={styles.live}>{lidarMeta.n.toLocaleString()} pts</span>
          <span className={thermalError ? styles.warn : styles.heat}>{thermalStatus}</span>
          <span className={styles.fusion}>{thermalConfirmed} thermal supports</span>
        </div>
      </div>
    </PanelShell>
  )
})

export default WorldMapPanel

function normalizeThermalCalibration(value) {
  const fallback = GUI_DEFAULT_THERMAL_CALIBRATION
  const intrValue = value?.intrinsics ?? value?.intr ?? {}
  const extrValue = value?.extrinsics ?? value?.extr ?? {}
  const intr = {
    width: intOr(intrValue.width, fallback.intr.width, 1),
    height: intOr(intrValue.height, fallback.intr.height, 1),
    fx: numberOr(intrValue.fx, fallback.intr.fx),
    fy: numberOr(intrValue.fy, fallback.intr.fy),
    cx: numberOr(intrValue.cx, fallback.intr.cx),
    cy: numberOr(intrValue.cy, fallback.intr.cy),
  }
  const extr = {
    tx: numberOr(extrValue.tx, fallback.extr.tx),
    ty: numberOr(extrValue.ty, fallback.extr.ty),
    tz: numberOr(extrValue.tz, fallback.extr.tz),
    rollDeg: numberOr(extrValue.rollDeg ?? extrValue.roll_deg, fallback.extr.rollDeg),
    pitchDeg: numberOr(extrValue.pitchDeg ?? extrValue.pitch_deg, fallback.extr.pitchDeg),
    yawDeg: numberOr(extrValue.yawDeg ?? extrValue.yaw_deg, fallback.extr.yawDeg),
  }
  return {
    intr,
    extr,
    source: String(value?.source ?? fallback.source),
  }
}

function numberOr(value, fallback) {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

function intOr(value, fallback, minValue) {
  return Math.max(minValue, Math.round(numberOr(value, fallback)))
}

function drawDetectionBoxes(ctx, boxes, selectedObjectKey) {
  disposeGroup(ctx.boxGroup)
  const viewport = ctx.renderer.getSize(new THREE.Vector2())
  boxes.forEach(box => {
    const pts = orientedBoxLines(box)
    const color = detectionColor(box.label, box.score)
    const selected = box.key === selectedObjectKey
    const line = selected
      ? makeWideBoxLine(pts, color, viewport)
      : makeThinBoxLine(pts, color)
    ctx.boxGroup.add(line)
  })
}

function makeThinBoxLine(points, color) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(p[0], p[1], p[2])))
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.95,
  })
  return new THREE.LineSegments(geometry, material)
}

function makeWideBoxLine(points, color, viewport) {
  const geometry = new LineSegmentsGeometry()
  geometry.setPositions(points.flat())
  const material = new LineMaterial({
    color,
    linewidth: 4,
    transparent: true,
    opacity: 1,
    depthTest: false,
  })
  material.resolution.set(Math.max(1, viewport.x), Math.max(1, viewport.y))
  const line = new LineSegments2(geometry, material)
  line.frustumCulled = false
  return line
}

function updateBoxMaterialResolution(group, width, height) {
  group.traverse(child => {
    if (child.material?.resolution?.set) {
      child.material.resolution.set(Math.max(1, width), Math.max(1, height))
    }
  })
}

function orientedBoxLines(box) {
  const [dx, dy, dz] = box.size
  const hx = Math.max(dx * 0.5, 0.05)
  const hy = Math.max(dy * 0.5, 0.05)
  const hz = Math.max(dz * 0.5, 0.05)
  const local = [
    [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
    [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
  ]
  const c = Math.cos(box.yaw)
  const s = Math.sin(box.yaw)
  const corners = local.map(([x, y, z]) => {
    const sensor = [
      c * x - s * y + box.center[0],
      s * x + c * y + box.center[1],
      z + box.center[2],
    ]
    return lidarToScene(sensor)
  })
  const edges = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]
  return edges.map(i => corners[i])
}

function lidarToScene(v) {
  // Ouster lidar is right-handed: +X forward, +Y left, +Z up.
  // Three.js is Y-up; map to +X right, +Y up, -Z forward without mirroring.
  return [-v[1], v[2], -v[0]]
}

function updateThermalProjection(ctx, frame, thermalFrame, n, calibration) {
  ctx.mat.uniforms.uHasThermal.value = thermalFrame?.data?.length ? 1 : 0
  if (!thermalFrame?.data?.length) {
    ctx.thermalValid.fill(0, 0, n)
    ctx.thermalValue.fill(0, 0, n)
    return
  }

  const { intr } = calibration
  const scaleU = thermalFrame.w / intr.width
  const scaleV = thermalFrame.h / intr.height

  for (let i = 0; i < n; i++) {
    const src = i * 3
    const projected = projectLidarToThermal(
      frame.positions[src],
      frame.positions[src + 1],
      frame.positions[src + 2],
      calibration,
    )

    if (!projected.valid) {
      ctx.thermalValid[i] = 0
      ctx.thermalValue[i] = 0
      continue
    }

    const u = Math.max(0, Math.min(thermalFrame.w - 1, Math.floor(projected.u * scaleU)))
    const v = Math.max(0, Math.min(thermalFrame.h - 1, Math.floor(projected.v * scaleV)))
    ctx.thermalValid[i] = 1
    ctx.thermalValue[i] = (thermalFrame.data[v * thermalFrame.w + u] ?? 0) / 255
  }
}

function updateScalarBounds(ctx, frame, n) {
  const ranges = sampledValues(n, i => {
    const src = i * 3
    const x = frame.positions[src]
    const y = frame.positions[src + 1]
    const z = frame.positions[src + 2]
    return Math.hypot(x, y, z)
  })
  const heights = sampledValues(n, i => frame.positions[i * 3 + 2])
  const intensities = sampledValues(n, i => frame.intensities[i])
  const [rangeLo, rangeHi] = percentileBounds(ranges, 2, 98, [0, 30])
  const [heightLo, heightHi] = percentileBounds(heights, 2, 98, [-1, 3])
  const [intensityLo, intensityHi] = percentileBounds(intensities, 2, 98, [0, 1])

  ctx.mat.uniforms.uRangeLo.value = rangeLo
  ctx.mat.uniforms.uRangeHi.value = rangeHi
  ctx.mat.uniforms.uHeightLo.value = heightLo
  ctx.mat.uniforms.uHeightHi.value = heightHi
  ctx.mat.uniforms.uIntensityLo.value = intensityLo
  ctx.mat.uniforms.uIntensityHi.value = intensityHi
}

function sampledValues(n, sampleFn) {
  const sampleCount = Math.min(n, 8192)
  const stride = Math.max(1, Math.floor(n / sampleCount))
  const values = []
  for (let i = 0; i < n; i += stride) {
    const value = sampleFn(i)
    if (Number.isFinite(value)) values.push(value)
  }
  values.sort((a, b) => a - b)
  return values
}

function percentileBounds(sortedValues, loPct, hiPct, fallback) {
  if (!sortedValues.length) return fallback
  const lo = sortedValues[Math.floor((sortedValues.length - 1) * loPct / 100)]
  let hi = sortedValues[Math.ceil((sortedValues.length - 1) * hiPct / 100)]
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return fallback
  if (hi - lo < 1e-6) hi = lo + 1
  return [lo, hi]
}

function projectLidarToThermal(x, y, z, calibration) {
  const { intr, extr } = calibration
  const r = lidarToCameraRotation(calibration)
  const camX = r[0][0] * x + r[0][1] * y + r[0][2] * z + extr.tx
  const camY = r[1][0] * x + r[1][1] * y + r[1][2] * z + extr.ty
  const camZ = r[2][0] * x + r[2][1] * y + r[2][2] * z + extr.tz

  if (camZ <= 0.05) return { valid: false, u: -1, v: -1 }
  const u = intr.fx * camX / camZ + intr.cx
  const v = intr.fy * camY / camZ + intr.cy
  return {
    valid: u >= 0 && u < intr.width && v >= 0 && v < intr.height,
    u,
    v,
  }
}

function makeThermalFovGroup(depthM, calibration) {
  const group = new THREE.Group()
  const { apex, near, far } = thermalFrustumPoints(depthM, calibration)
  const lineVertices = []

  far.forEach(corner => lineVertices.push(apex, corner))
  for (let i = 0; i < 4; i++) lineVertices.push(far[i], far[(i + 1) % 4])
  for (let i = 0; i < 4; i++) lineVertices.push(near[i], near[(i + 1) % 4])

  const lineGeometry = new THREE.BufferGeometry().setFromPoints(
    lineVertices.map(p => {
      const [x, y, z] = lidarToScene(p)
      return new THREE.Vector3(x, y, z)
    })
  )
  group.add(new THREE.LineSegments(
    lineGeometry,
    new THREE.LineBasicMaterial({ color: 0xffcc00, transparent: true, opacity: 0.94 })
  ))

  const meshPoints = []
  for (let i = 0; i < 4; i++) {
    meshPoints.push(apex, far[i], far[(i + 1) % 4])
  }
  const meshPositions = new Float32Array(meshPoints.flatMap(p => lidarToScene(p)))
  const meshGeometry = new THREE.BufferGeometry()
  meshGeometry.setAttribute('position', new THREE.BufferAttribute(meshPositions, 3))
  meshGeometry.computeVertexNormals()
  group.add(new THREE.Mesh(
    meshGeometry,
    new THREE.MeshBasicMaterial({
      color: 0xffcc00,
      transparent: true,
      opacity: 0.075,
      side: THREE.DoubleSide,
      depthWrite: false,
    })
  ))

  return group
}

function thermalFrustumPoints(depthM, calibration) {
  const { intr, extr } = calibration
  const nearM = Math.max(0.05, depthM * 0.05)
  const farM = Math.max(nearM + 0.01, depthM)
  const cornersAt = (d) => {
    const ux = [0, intr.width, intr.width, 0].map(u => (u - intr.cx) / intr.fx)
    const vy = [0, 0, intr.height, intr.height].map(v => (v - intr.cy) / intr.fy)
    return ux.map((x, i) => [x * d, vy[i] * d, d])
  }
  return {
    apex: cameraToLidar([0, 0, 0], extr, calibration),
    near: cornersAt(nearM).map(p => cameraToLidar(p, extr, calibration)),
    far: cornersAt(farM).map(p => cameraToLidar(p, extr, calibration)),
  }
}

function cameraToLidar(cam, extr, calibration) {
  const r = lidarToCameraRotation(calibration)
  const p = [cam[0] - extr.tx, cam[1] - extr.ty, cam[2] - extr.tz]
  return [
    r[0][0] * p[0] + r[1][0] * p[1] + r[2][0] * p[2],
    r[0][1] * p[0] + r[1][1] * p[1] + r[2][1] * p[2],
    r[0][2] * p[0] + r[1][2] * p[1] + r[2][2] * p[2],
  ]
}

function lidarToCameraRotation(calibration) {
  const { extr } = calibration
  const rUser = eulerToR(
    THREE.MathUtils.degToRad(extr.rollDeg),
    THREE.MathUtils.degToRad(extr.pitchDeg),
    THREE.MathUtils.degToRad(extr.yawDeg),
  )
  const rBase = [
    [0, -1, 0],
    [0, 0, -1],
    [1, 0, 0],
  ]
  return matMul3(rUser, rBase)
}

function eulerToR(roll, pitch, yaw) {
  const cr = Math.cos(roll), sr = Math.sin(roll)
  const cp = Math.cos(pitch), sp = Math.sin(pitch)
  const cy = Math.cos(yaw), sy = Math.sin(yaw)
  const rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
  const ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
  const rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
  return matMul3(matMul3(rz, ry), rx)
}

function matMul3(a, b) {
  const out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
    }
  }
  return out
}

function alejandroCameraPosition() {
  const elevation = THREE.MathUtils.degToRad(VIEW_ELEVATION_DEG)
  const azimuth = THREE.MathUtils.degToRad(VIEW_AZIMUTH_DEG)
  const nativeX = Math.cos(elevation) * Math.cos(azimuth) * VIEW_DISTANCE
  const nativeY = Math.cos(elevation) * Math.sin(azimuth) * VIEW_DISTANCE
  const nativeZ = Math.sin(elevation) * VIEW_DISTANCE
  const [x, y, z] = lidarToScene([nativeX, nativeY, nativeZ])
  return { x, y, z }
}

function makeAlejandroGrid() {
  const grid = new THREE.GridHelper(40, 20, 0x2d2d34, 0x1a1a20)
  grid.material.transparent = true
  grid.material.opacity = 0.72
  return grid
}

function makeLidarAxes() {
  const group = new THREE.Group()
  const axes = [
    { end: lidarToScene([1.5, 0, 0]), color: 0xff2222 },
    { end: lidarToScene([0, 1.5, 0]), color: 0x22dd44 },
    { end: lidarToScene([0, 0, 1.5]), color: 0x2277ff },
  ]

  axes.forEach(axis => {
    const points = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(axis.end[0], axis.end[1], axis.end[2]),
    ]
    group.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color: axis.color, linewidth: 2 })
    ))
  })

  return group
}

function normalizeDetections(items) {
  if (!Array.isArray(items)) return []
  return items.map((item, i) => {
    const center = vec3(item.center ?? item.centroid ?? item.position, [0, 0, 0])
    const size = vec3(item.size, null)
      ?? bboxSize(item.bbox_min ?? item.min, item.bbox_max ?? item.max)
      ?? [0.35, 0.35, 0.8]
    const id = item.track_id ?? item.id ?? i + 1
    const source = item.source ?? 'detector'
    return {
      id,
      key: objectKey(source, id),
      source,
      center,
      size,
      yaw: Number(item.yaw ?? item.heading ?? 0) || 0,
      label: String(item.class_name ?? item.label ?? item.kind ?? 'object').replaceAll('_', ' '),
      score: clamp01(item.fusion_score ?? item.score ?? item.confidence ?? item.model_score ?? 0),
      modelScore: clamp01(item.model_score ?? item.score ?? 0),
      thermalScore: clamp01(item.thermal_score ?? 0),
      thermalCoverage: clamp01(item.thermal_coverage ?? 0),
      fusionNote: String(item.fusion_note ?? ''),
    }
  }).filter(item => item.size.every(Number.isFinite) && item.center.every(Number.isFinite))
}

function objectKey(source, id) {
  return `${String(source)}:${String(id)}`
}

function vec3(value, fallback) {
  if (Array.isArray(value) && value.length >= 3) return value.slice(0, 3).map(Number)
  if (value && typeof value === 'object') {
    return [Number(value.x), Number(value.y), Number(value.z ?? value.height ?? 0)]
  }
  return fallback
}

function bboxSize(minValue, maxValue) {
  const mn = vec3(minValue, null)
  const mx = vec3(maxValue, null)
  if (!mn || !mx) return null
  return [Math.abs(mx[0] - mn[0]), Math.abs(mx[1] - mn[1]), Math.abs(mx[2] - mn[2])]
}

function detectionColor(label, score) {
  const normalized = label.toLowerCase()
  if (normalized.includes('occupied chair')) return 0xffb800
  if (normalized.includes('chair')) return 0xff9e1f
  if (normalized.includes('seated')) return 0xff611f
  if (normalized.includes('standing') || normalized.includes('person') || normalized.includes('human')) return 0xff2947
  if (normalized.includes('cyclist')) return 0x00d9ff
  if (normalized.includes('car') || normalized.includes('vehicle')) return 0xff9e1f
  return confidenceColor(score)
}

function disposeGroup(group) {
  while (group.children.length) {
    const child = group.children.pop()
    child.geometry?.dispose?.()
    child.material?.dispose?.()
  }
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
