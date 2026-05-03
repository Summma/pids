import { useState, useEffect, useRef, useCallback } from 'react'
import { useWebSocket } from './useWebSocket'
import { wsUrl, WS_PATHS } from '@/utils/wsConfig'

const RING_SIZE = 12

export function useThermal(streamConfig) {
  const { connState, setOnMessage, send } = useWebSocket(wsUrl(WS_PATHS.thermal, {}, streamConfig))
  const frameRef = useRef(null)   // { data: Uint8Array, w, h }
  const ringRef = useRef([])
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
        const next = {
          data: arr,
          w: env.w ?? 640,
          h: env.h ?? 512,
          calibration: env.calibration ?? null,
          ts: env.ts ?? 0,
          seq: env.seq ?? 0,
        }
        frameRef.current = next
        const ring = ringRef.current
        ring.push(next)
        if (ring.length > RING_SIZE) ring.shift()
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

  const frameAt = useCallback((targetTs) => closestByTs(ringRef.current, targetTs), [])

  const sendControl = useCallback((cmd) => send(JSON.stringify(cmd)), [send])

  return { connState, frameRef, frameAt, meta, error, sendControl }
}

function closestByTs(ring, targetTs) {
  if (!ring.length) return null
  if (!Number.isFinite(targetTs) || targetTs <= 0) return ring[ring.length - 1]
  let best = ring[0]
  let bestDt = Math.abs(best.ts - targetTs)
  for (let i = 1; i < ring.length; i++) {
    const dt = Math.abs(ring[i].ts - targetTs)
    if (dt < bestDt) {
      bestDt = dt
      best = ring[i]
    }
  }
  return best
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
