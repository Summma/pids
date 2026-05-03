import { useState, useEffect } from 'react'
import { useWebSocket } from './useWebSocket'
import { wsUrl, WS_PATHS } from '@/utils/wsConfig'

export function useCamera() {
  const { connState, setOnMessage, send } = useWebSocket(wsUrl(WS_PATHS.camera))
  const [frame, setFrame] = useState({ src: '', w: 0, h: 0, seq: 0, ts: 0 })
  const [error, setError] = useState('')

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
        setFrame({
          src: `data:${env.mime ?? 'image/jpeg'};base64,${env.data}`,
          w: env.w ?? 0,
          h: env.h ?? 0,
          seq: env.seq ?? 0,
          ts: env.ts ?? 0,
        })
      } catch {}
    })
  }, [setOnMessage])

  const sendControl = (cmd) => send(JSON.stringify(cmd))

  return { connState, frame, error, sendControl }
}
