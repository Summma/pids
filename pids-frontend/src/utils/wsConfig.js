const stored = (() => {
  try { return JSON.parse(localStorage.getItem('pids_cfg')) ?? {} } catch { return {} }
})()

const defaultHost = window.location.hostname || 'localhost'
const envHost = import.meta.env.VITE_PIDS_HOST
const envPort = import.meta.env.VITE_PIDS_PORT
const envDetectionMode = import.meta.env.VITE_PIDS_DETECTION_MODE
const envLidarMaxPoints = import.meta.env.VITE_PIDS_LIDAR_MAX_POINTS

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

export let HOST = envHost ?? stored.host ?? defaultHost
export let PORT = Number(envPort ?? stored.port ?? 9090)
export let DETECTION_MODE = normalizeDetectionMode(envDetectionMode ?? stored.detectionMode ?? 'auto')
export let LIDAR_MAX_POINTS = normalizeLidarMaxPoints(envLidarMaxPoints ?? stored.lidarMaxPoints ?? 60000)

export function getWsBase() { return `ws://${HOST}:${PORT}` }
export function getHttpBase() {
  const proto = window.location.protocol === 'https:' ? 'https' : 'http'
  return `${proto}://${HOST}:${PORT}`
}

export const WS_PATHS = {
  thermal: '/thermal',
  lidar:   '/lidar',
  camera:  '/camera',
  rf:      '/rf',
  threats: '/threats',
  fusion:  '/fusion',
}

export function wsUrl(path, params = {}) {
  const query = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') query.set(key, value)
  })
  const suffix = query.toString()
  return `${getWsBase()}${path}${suffix ? `?${suffix}` : ''}`
}

export function apiUrl(path) { return `${getHttpBase()}${path}` }

export function saveConfig(host, port, detectionMode = DETECTION_MODE, lidarMaxPoints = LIDAR_MAX_POINTS) {
  const normalizedDetectionMode = normalizeDetectionMode(detectionMode)
  const normalizedLidarMaxPoints = normalizeLidarMaxPoints(lidarMaxPoints)
  HOST = host
  PORT = port
  DETECTION_MODE = normalizedDetectionMode
  LIDAR_MAX_POINTS = normalizedLidarMaxPoints
  try {
    localStorage.setItem('pids_cfg', JSON.stringify({
      host,
      port,
      detectionMode: normalizedDetectionMode,
      lidarMaxPoints: normalizedLidarMaxPoints,
    }))
  } catch {}
}

export function normalizeDetectionMode(value) {
  const mode = String(value ?? '').trim()
  return DETECTION_MODE_OPTIONS.some(option => option.value === mode) ? mode : 'auto'
}

export function normalizeLidarMaxPoints(value) {
  const points = Number(value)
  if (!Number.isFinite(points)) return 60000
  const rounded = Math.round(points)
  const clamped = Math.max(1000, Math.min(rounded, 131072))
  return LIDAR_POINT_OPTIONS.some(option => option.value === clamped) ? clamped : 60000
}
