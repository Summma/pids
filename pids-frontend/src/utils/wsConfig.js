const stored = (() => {
  try { return JSON.parse(localStorage.getItem('pids_cfg')) ?? {} } catch { return {} }
})()

const defaultHost = window.location.hostname || 'localhost'
const envHost = import.meta.env.VITE_PIDS_HOST
const envPort = import.meta.env.VITE_PIDS_PORT

export let HOST = envHost ?? stored.host ?? defaultHost
export let PORT = Number(envPort ?? stored.port ?? 9090)

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

export function wsUrl(path) { return `${getWsBase()}${path}` }
export function apiUrl(path) { return `${getHttpBase()}${path}` }

export function saveConfig(host, port) {
  HOST = host
  PORT = port
  try { localStorage.setItem('pids_cfg', JSON.stringify({ host, port })) } catch {}
}
