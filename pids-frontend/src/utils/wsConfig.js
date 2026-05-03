const stored = (() => {
  try { return JSON.parse(localStorage.getItem('pids_cfg')) ?? {} } catch { return {} }
})()

const defaultHost = window.location.hostname || 'localhost'
const envHost = import.meta.env.VITE_PIDS_HOST
const envPort = import.meta.env.VITE_PIDS_PORT
const envDetectionMode = import.meta.env.VITE_PIDS_DETECTION_MODE
const envLidarMaxPoints = import.meta.env.VITE_PIDS_LIDAR_MAX_POINTS
const envThermalPalette = import.meta.env.VITE_PIDS_THERMAL_PALETTE
const envShowThermalFov = import.meta.env.VITE_PIDS_SHOW_THERMAL_FOV

export const DETECTION_MODE_OPTIONS = [
  {
    value: 'auto',
    label: 'Auto fusion',
    title: 'Indoor ROI detector with thermal fusion and PointPillars support',
  },
  {
    value: 'indoor_human',
    label: 'Indoor ROI',
    title: 'Geometry-based indoor seated/standing candidate detector with thermal fusion',
  },
  {
    value: 'roi_thermal',
    label: 'Thermal ROI',
    title: 'Indoor ROI detector gated by projected thermal evidence to reduce furniture false positives',
  },
  {
    value: 'seated_roi',
    label: 'Seated ROI',
    title: 'Seated-person and occupied-chair ROI profile with denser clustering and thermal support',
  },
  {
    value: 'pointnet_roi',
    label: 'PointNet++ ROI + thermal',
    title: 'Thermal-aware crop profile shaped for PointNet++/PointNet-style ROI classification',
  },
  {
    value: 'pointnext_roi',
    label: 'PointNeXt ROI + thermal',
    title: 'Thermal-aware crop profile shaped for PointNeXt-style seated/person ROI classification',
  },
  {
    value: 'dgcnn_roi',
    label: 'DGCNN ROI + thermal',
    title: 'Thermal-aware crop profile emphasizing local edge/shape evidence like DGCNN EdgeConv',
  },
  {
    value: 'kpconv_roi',
    label: 'KPConv ROI + thermal',
    title: 'Thermal-aware crop profile emphasizing geometric support like kernel point convolutions',
  },
  {
    value: 'sparse_cnn_roi',
    label: 'Sparse CNN ROI + thermal',
    title: 'Thermal-aware crop profile emphasizing voxel occupancy support for sparse CNN models',
  },
  {
    value: 'point_transformer_roi',
    label: 'Point Transformer ROI + thermal',
    title: 'Thermal-aware crop profile emphasizing context and heat evidence for transformer-style models',
  },
  {
    value: 'pointpillars',
    label: 'PointPillars',
    title: 'Jetson PointPillars detector with thermal fusion',
  },
  {
    value: 'off',
    label: 'Off',
    title: 'Stream lidar without object detections',
  },
]

export const LIDAR_POINT_OPTIONS = [
  { value: 30000, label: '30,000' },
  { value: 60000, label: '60,000' },
  { value: 90000, label: '90,000' },
  { value: 120000, label: '120,000' },
]

export const THERMAL_PALETTE_OPTIONS = [
  { value: 'iron', label: 'Iron' },
  { value: 'white_hot', label: 'White hot' },
  { value: 'black_hot', label: 'Black hot' },
  { value: 'turbo', label: 'Turbo' },
]

export let HOST = envHost ?? stored.host ?? defaultHost
export let PORT = Number(envPort ?? stored.port ?? 9090)
export let DETECTION_MODE = normalizeDetectionMode(envDetectionMode ?? stored.detectionMode ?? 'auto')
export let LIDAR_MAX_POINTS = normalizeLidarMaxPoints(envLidarMaxPoints ?? stored.lidarMaxPoints ?? 30000)
export let THERMAL_PALETTE = normalizeThermalPalette(envThermalPalette ?? stored.thermalPalette ?? 'iron')
export let SHOW_THERMAL_FOV = normalizeBool(envShowThermalFov ?? stored.showThermalFov ?? true)

export function getWsBase(config = {}) {
  const host = config.host ?? HOST
  const port = config.port ?? PORT
  return `ws://${host}:${port}`
}

export function getHttpBase(config = {}) {
  const proto = window.location.protocol === 'https:' ? 'https' : 'http'
  const host = config.host ?? HOST
  const port = config.port ?? PORT
  return `${proto}://${host}:${port}`
}

export const WS_PATHS = {
  thermal: '/thermal',
  lidar:   '/lidar',
  camera:  '/camera',
  rf:      '/rf',
  threats: '/threats',
  fusion:  '/fusion',
}

export function wsUrl(path, params = {}, config = {}) {
  const query = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') query.set(key, value)
  })
  const suffix = query.toString()
  return `${getWsBase(config)}${path}${suffix ? `?${suffix}` : ''}`
}

export function apiUrl(path, config = {}) { return `${getHttpBase(config)}${path}` }

export function saveConfig(
  host,
  port,
  detectionMode = DETECTION_MODE,
  lidarMaxPoints = LIDAR_MAX_POINTS,
  thermalPalette = THERMAL_PALETTE,
  showThermalFov = SHOW_THERMAL_FOV,
) {
  const normalizedDetectionMode = normalizeDetectionMode(detectionMode)
  const normalizedLidarMaxPoints = normalizeLidarMaxPoints(lidarMaxPoints)
  const normalizedThermalPalette = normalizeThermalPalette(thermalPalette)
  const normalizedShowThermalFov = normalizeBool(showThermalFov)
  HOST = host
  PORT = port
  DETECTION_MODE = normalizedDetectionMode
  LIDAR_MAX_POINTS = normalizedLidarMaxPoints
  THERMAL_PALETTE = normalizedThermalPalette
  SHOW_THERMAL_FOV = normalizedShowThermalFov
  try {
    localStorage.setItem('pids_cfg', JSON.stringify({
      host,
      port,
      detectionMode: normalizedDetectionMode,
      lidarMaxPoints: normalizedLidarMaxPoints,
      thermalPalette: normalizedThermalPalette,
      showThermalFov: normalizedShowThermalFov,
    }))
  } catch {}

  return {
    host: HOST,
    port: PORT,
    detectionMode: DETECTION_MODE,
    lidarMaxPoints: LIDAR_MAX_POINTS,
    thermalPalette: THERMAL_PALETTE,
    showThermalFov: SHOW_THERMAL_FOV,
  }
}

export function normalizeDetectionMode(value) {
  const mode = String(value ?? '').trim()
  return DETECTION_MODE_OPTIONS.some(option => option.value === mode) ? mode : 'auto'
}

export function normalizeLidarMaxPoints(value) {
  const points = Number(value)
  if (!Number.isFinite(points)) return 30000
  const rounded = Math.round(points)
  const clamped = Math.max(1000, Math.min(rounded, 131072))
  return LIDAR_POINT_OPTIONS.some(option => option.value === clamped) ? clamped : 30000
}

export function normalizeThermalPalette(value) {
  const palette = String(value ?? '').trim()
  return THERMAL_PALETTE_OPTIONS.some(option => option.value === palette) ? palette : 'iron'
}

export function normalizeBool(value) {
  if (typeof value === 'boolean') return value
  const text = String(value ?? '').trim().toLowerCase()
  if (['1', 'true', 'yes', 'on'].includes(text)) return true
  if (['0', 'false', 'no', 'off'].includes(text)) return false
  return Boolean(value)
}
