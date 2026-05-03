const stored = (() => {
  try { return JSON.parse(localStorage.getItem('pids_cfg')) ?? {} } catch { return {} }
})()

const defaultHost = window.location.hostname || 'localhost'
const envHost = import.meta.env.VITE_PIDS_HOST
const envPort = import.meta.env.VITE_PIDS_PORT
const envDetectionMode = import.meta.env.VITE_PIDS_DETECTION_MODE

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

export let HOST = envHost ?? stored.host ?? defaultHost
export let PORT = Number(envPort ?? stored.port ?? 9090)
export let DETECTION_MODE = normalizeDetectionMode(envDetectionMode ?? stored.detectionMode ?? 'auto')

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

export function saveConfig(host, port, detectionMode = DETECTION_MODE) {
  const normalizedDetectionMode = normalizeDetectionMode(detectionMode)
  HOST = host
  PORT = port
  DETECTION_MODE = normalizedDetectionMode
  try {
    localStorage.setItem('pids_cfg', JSON.stringify({
      host,
      port,
      detectionMode: normalizedDetectionMode,
    }))
  } catch {}
}

export function normalizeDetectionMode(value) {
  const mode = String(value ?? '').trim()
  return DETECTION_MODE_OPTIONS.some(option => option.value === mode) ? mode : 'auto'
}
