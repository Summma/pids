// Synthesize "person" detection objects from YOLO bboxes by sampling the
// thermal frame at the bbox center (for temperature) and the lidar cloud
// at the bbox center (for 3D position + range). Assumes the visible camera,
// thermal camera, and lidar share boresight (same scene FOV).

const WEBCAM_HFOV_DEG = 55
const R_LIDAR_TO_CAM_BASE = [
  [0, -1, 0],
  [0, 0, -1],
  [1, 0, 0],
]

export function synthesizePersonDetections({
  persons,
  frameW,
  frameH,
  thermalFrame,
  thermalMin,
  thermalMax,
  thermalCalibration,
  lidarFrame,
}) {
  if (!Array.isArray(persons) || persons.length === 0) return []
  if (!frameW || !frameH) return []
  const cal = normalizeCalibration(thermalCalibration)
  if (!cal) return []

  const { intr } = cal
  const thermalHfov = 2 * Math.atan(intr.width / (2 * intr.fx))
  const thermalVfov = 2 * Math.atan(intr.height / (2 * intr.fy))
  const webcamHfov = degToRad(WEBCAM_HFOV_DEG)
  const webcamFocalPx = (frameW / 2) / Math.tan(webcamHfov / 2)
  const cropW = Math.min(frameW, 2 * webcamFocalPx * Math.tan(thermalHfov / 2))
  const cropH = Math.min(frameH, 2 * webcamFocalPx * Math.tan(thermalVfov / 2))
  const sx = (frameW - cropW) / 2
  const sy = (frameH - cropH) / 2

  const projected = projectLidarToThermal(lidarFrame, cal)

  return persons.map((person, i) => {
    const bbox = person?.bbox_xyxy
    if (!Array.isArray(bbox) || bbox.length < 4) return null

    // bbox center in webcam pixel coords -> thermal pixel coords
    const cxW = (bbox[0] + bbox[2]) / 2
    const cyW = (bbox[1] + bbox[3]) / 2
    const cxT = clamp((cxW - sx) / cropW, 0, 1) * intr.width
    const cyT = clamp((cyW - sy) / cropH, 0, 1) * intr.height
    const tx1 = clamp((bbox[0] - sx) / cropW, 0, 1) * intr.width
    const ty1 = clamp((bbox[1] - sy) / cropH, 0, 1) * intr.height
    const tx2 = clamp((bbox[2] - sx) / cropW, 0, 1) * intr.width
    const ty2 = clamp((bbox[3] - sy) / cropH, 0, 1) * intr.height

    const thermal = sampleThermalAt(thermalFrame, thermalMin, thermalMax, cxT, cyT)
    const lidar = sampleLidarNear(projected, cxT, cyT)

    const center = lidar?.position ?? [0, 0, 0]
    const range = lidar?.range ?? 0

    return {
      track_id: `yolo-${i}`,
      source: 'yolo',
      class_name: 'person',
      score: Number(person.confidence ?? 0),
      model_score: Number(person.confidence ?? 0),
      fusion_score: Number(person.confidence ?? 0),
      center,
      size: [
        Math.max(0.3, Math.abs(tx2 - tx1) / 100),
        Math.max(0.3, Math.abs(tx2 - tx1) / 100),
        Math.max(0.6, Math.abs(ty2 - ty1) / 100),
      ],
      range,
      thermal_score: thermal?.normalized ?? 0,
      thermal_max: thermal?.normalized ?? 0,
      thermal_coverage: thermal ? 1 : 0,
      thermal_temp_c: thermal?.temperatureC,
      fusion_note: lidar ? 'yolo·person' : 'yolo·person·no_lidar',
      yaw: 0,
    }
  }).filter(Boolean)
}

function sampleThermalAt(thermalFrame, tMin, tMax, u, v) {
  if (!thermalFrame?.data || !thermalFrame.w || !thermalFrame.h) return null
  const x = Math.trunc(u)
  const y = Math.trunc(v)
  if (x < 0 || y < 0 || x >= thermalFrame.w || y >= thermalFrame.h) return null
  const idx = y * thermalFrame.w + x
  if (idx < 0 || idx >= thermalFrame.data.length) return null
  const palette = thermalFrame.data[idx]
  const lo = Number.isFinite(tMin) ? tMin : 0
  const hi = Number.isFinite(tMax) && tMax > lo ? tMax : lo + 1
  return {
    normalized: palette / 255,
    temperatureC: lo + (palette / 255) * (hi - lo),
  }
}

function projectLidarToThermal(lidarFrame, calibration) {
  const points = lidarFrame?.n ?? 0
  if (!lidarFrame?.positions || points <= 0) return null

  const { intr, extr } = calibration
  const r = lidarToCameraRotation(calibration)
  const us = new Float32Array(points)
  const vs = new Float32Array(points)
  const ranges = new Float32Array(points)
  const positions = lidarFrame.positions
  const xs = new Float32Array(points)
  const ys = new Float32Array(points)
  const zs = new Float32Array(points)
  let valid = 0

  for (let i = 0; i < points; i++) {
    const src = i * 3
    const x = positions[src]
    const y = positions[src + 1]
    const z = positions[src + 2]
    const range = Math.hypot(x, y, z)
    if (!Number.isFinite(range) || range <= 0.3) continue

    const camX = r[0][0] * x + r[0][1] * y + r[0][2] * z + extr.tx
    const camY = r[1][0] * x + r[1][1] * y + r[1][2] * z + extr.ty
    const camZ = r[2][0] * x + r[2][1] * y + r[2][2] * z + extr.tz
    if (camZ <= 0.05) continue

    const u = intr.fx * camX / camZ + intr.cx
    const v = intr.fy * camY / camZ + intr.cy
    if (u < 0 || u >= intr.width || v < 0 || v >= intr.height) continue

    us[valid] = u
    vs[valid] = v
    ranges[valid] = range
    xs[valid] = x
    ys[valid] = y
    zs[valid] = z
    valid++
  }
  return { us, vs, ranges, xs, ys, zs, n: valid }
}

function sampleLidarNear(projected, cxT, cyT) {
  if (!projected || projected.n === 0) return null
  let bestIdx = -1
  let bestDist = Infinity
  for (let i = 0; i < projected.n; i++) {
    const du = projected.us[i] - cxT
    const dv = projected.vs[i] - cyT
    const d = du * du + dv * dv
    if (d < bestDist) {
      bestDist = d
      bestIdx = i
    }
  }
  if (bestIdx < 0) return null
  // Reject if the nearest projected point is further than ~24 thermal pixels
  // away from the bbox center; the bbox is then probably not seeing anything
  // the lidar can range.
  if (Math.sqrt(bestDist) > 24) return null
  return {
    range: projected.ranges[bestIdx],
    position: [projected.xs[bestIdx], projected.ys[bestIdx], projected.zs[bestIdx]],
  }
}

function normalizeCalibration(value) {
  if (!value || typeof value !== 'object') return null
  const intrValue = value.intrinsics ?? value.intr ?? {}
  const extrValue = value.extrinsics ?? value.extr ?? {}
  const intr = {
    width: numberOr(intrValue.width, 0),
    height: numberOr(intrValue.height, 0),
    fx: numberOr(intrValue.fx, 0),
    fy: numberOr(intrValue.fy, 0),
    cx: numberOr(intrValue.cx, 0),
    cy: numberOr(intrValue.cy, 0),
  }
  if (intr.width <= 0 || intr.height <= 0 || intr.fx <= 0 || intr.fy <= 0) return null
  const extr = {
    tx: numberOr(extrValue.tx, 0),
    ty: numberOr(extrValue.ty, 0),
    tz: numberOr(extrValue.tz, 0),
    rollDeg: numberOr(extrValue.rollDeg ?? extrValue.roll_deg, 0),
    pitchDeg: numberOr(extrValue.pitchDeg ?? extrValue.pitch_deg, 0),
    yawDeg: numberOr(extrValue.yawDeg ?? extrValue.yaw_deg, 0),
  }
  return { intr, extr }
}

function lidarToCameraRotation(calibration) {
  const { extr } = calibration
  return matMul3(
    eulerToR(degToRad(extr.rollDeg), degToRad(extr.pitchDeg), degToRad(extr.yawDeg)),
    R_LIDAR_TO_CAM_BASE,
  )
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

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v))
}

function degToRad(v) {
  return v * Math.PI / 180
}

function numberOr(value, fallback) {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}
