import { useState, useEffect, useRef, useCallback } from 'react'
import { useWebSocket } from './useWebSocket'
import { wsUrl, WS_PATHS } from '@/utils/wsConfig'

export function useThermal(streamConfig) {
  const { connState, setOnMessage, send } = useWebSocket(wsUrl(WS_PATHS.thermal, {}, streamConfig))
  const frameRef = useRef(null)   // { data: Uint8Array, w, h }
  const [meta, setMeta] = useState({ tMin: 0, tMax: 100, seq: 0, ts: 0, w: 640, h: 512, calibration: null })
  const [error, setError] = useState('')

  useEffect(() => {
    setOnMessage((e) => {
      try {
        const env = JSON.parse(e.data)
        if (env.type === 'calibration') {
          setMeta(prev => ({ ...prev, calibration: nextCalibration(prev.calibration, env.calibration), seq: env.seq ?? prev.seq, ts: env.ts ?? prev.ts }))
          return
        }
        if (env.type === 'error') {
          frameRef.current = null
          setMeta(prev => ({ ...prev, calibration: nextCalibration(prev.calibration, env.calibration) }))
          setError(env.message ?? 'thermal stream unavailable')
          return
        }
        if (env.type !== 'frame') return
        setError('')
        const raw = atob(env.data)
        const arr = new Uint8Array(raw.length)
        for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i)
        frameRef.current = { data: arr, w: env.w ?? 640, h: env.h ?? 512, calibration: env.calibration ?? null }
        setMeta(prev => ({
          tMin: env.t_min,
          tMax: env.t_max,
          seq: env.seq,
          ts: env.ts,
          w: env.w ?? 640,
          h: env.h ?? 512,
          calibration: nextCalibration(prev.calibration, env.calibration),
        }))
      } catch {}
    })
  }, [setOnMessage])

  const sendControl = useCallback((cmd) => send(JSON.stringify(cmd)), [send])

  return { connState, frameRef, meta, error, sendControl }
}

function nextCalibration(previous, incoming) {
  if (!incoming || typeof incoming !== 'object') return previous
  if (sameCalibration(previous, incoming)) return previous
  return incoming
}

function sameCalibration(a, b) {
  if (!a || !b) return false
  try {
    return JSON.stringify(a) === JSON.stringify(b)
  } catch {
    return false
  }
}
