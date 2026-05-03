const stored = (() => {
  try { return JSON.parse(localStorage.getItem('pids_cfg')) ?? {} } catch { return {} }
})()

export let HOST = stored.host ?? '192.168.1.42'
export let PORT = stored.port ?? 9090

export function getWsBase() { return `ws://${HOST}:${PORT}` }

export const WS_PATHS = {
  thermal: '/thermal',
  lidar:   '/lidar',
  rf:      '/rf',
  threats: '/threats',
  fusion:  '/fusion',
}

export function wsUrl(path) { return `${getWsBase()}${path}` }

export function saveConfig(host, port) {
  HOST = host
  PORT = port
  try { localStorage.setItem('pids_cfg', JSON.stringify({ host, port })) } catch {}
}
