import { useState, useEffect, useRef, useCallback } from 'react'
import { useWebSocket } from './useWebSocket'
import { wsUrl, WS_PATHS } from '@/utils/wsConfig'

const RING_SIZE = 12

export function useCamera(streamConfig) {
  const { connState, setOnMessage, send } = useWebSocket(wsUrl(WS_PATHS.camera, {}, streamConfig))
  const [frame, setFrame] = useState({ src: '', w: 0, h: 0, frameW: 0, frameH: 0, seq: 0, ts: 0, persons: [] })
  const [error, setError] = useState('')
  const ringRef = useRef([])

  useEffect(() => {
    setOnMessage((e) => {
      try {
        const env = JSON.parse(e.data)
        if (env.type === 'error') {
          setError(env.message ?? 'camera stream unavailable')
          return
        }
        if (env.type !== 'frame') return
        setError('')
        const next = {
          src: `data:${env.mime ?? 'image/jpeg'};base64,${env.data}`,
          w: env.w ?? 0,
          h: env.h ?? 0,
          frameW: env.frame_w ?? env.w ?? 0,
          frameH: env.frame_h ?? env.h ?? 0,
          seq: env.seq ?? 0,
          ts: env.ts ?? 0,
          persons: Array.isArray(env.persons) ? env.persons : [],
        }
        const ring = ringRef.current
        ring.push(next)
        if (ring.length > RING_SIZE) ring.shift()
        setFrame(next)
      } catch {}
    })
  }, [setOnMessage])

  const frameAt = useCallback((targetTs) => closestByTs(ringRef.current, targetTs), [])

  const sendControl = (cmd) => send(JSON.stringify(cmd))

  return { connState, frame, frameAt, error, sendControl }
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
