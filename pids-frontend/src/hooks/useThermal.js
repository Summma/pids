import { useState, useEffect, useRef, useCallback } from 'react'
import { useWebSocket } from './useWebSocket'
import { wsUrl, WS_PATHS } from '@/utils/wsConfig'

export function useThermal() {
  const { connState, setOnMessage, send } = useWebSocket(wsUrl(WS_PATHS.thermal))
  const frameRef = useRef(null)   // { data: Uint8Array, w, h }
  const [meta, setMeta] = useState({ tMin: 0, tMax: 100, seq: 0, ts: 0, w: 640, h: 512 })
  const [error, setError] = useState('')

  useEffect(() => {
    setOnMessage((e) => {
      try {
        const env = JSON.parse(e.data)
        if (env.type === 'error') {
          frameRef.current = null
          setError(env.message ?? 'thermal stream unavailable')
          return
        }
        if (env.type !== 'frame') return
        setError('')
        const raw = atob(env.data)
        const arr = new Uint8Array(raw.length)
        for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i)
        frameRef.current = { data: arr, w: env.w ?? 640, h: env.h ?? 512 }
        setMeta({ tMin: env.t_min, tMax: env.t_max, seq: env.seq, ts: env.ts, w: env.w ?? 640, h: env.h ?? 512 })
      } catch {}
    })
  }, [setOnMessage])

  const sendControl = useCallback((cmd) => send(JSON.stringify(cmd)), [send])

  return { connState, frameRef, meta, error, sendControl }
}
