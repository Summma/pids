import { useState, useEffect, useRef } from 'react'
import { useWebSocket } from './useWebSocket'
import { wsUrl, WS_PATHS } from '@/utils/wsConfig'

const MAX_PTS   = 131072
const posBuf    = new Float32Array(MAX_PTS * 3)
const intBuf    = new Float32Array(MAX_PTS)
const rawBuf    = new Uint8Array(MAX_PTS * 16)

export function useLidar() {
  const { connState, setOnMessage, send } = useWebSocket(wsUrl(WS_PATHS.lidar))
  const frameRef = useRef({ positions: posBuf, intensities: intBuf, n: 0 })
  const [meta, setMeta] = useState({ n: 0, seq: 0, ts: 0 })
  const [clusters, setClusters] = useState([])

  useEffect(() => {
    setOnMessage((e) => {
      try {
        const env = JSON.parse(e.data)
        if (env.type !== 'frame') return

        const str = atob(env.data)
        const len = Math.min(str.length, rawBuf.length)
        for (let i = 0; i < len; i++) rawBuf[i] = str.charCodeAt(i)

        const view = new DataView(rawBuf.buffer, 0, len)
        const n = Math.min(env.n, MAX_PTS)
        for (let i = 0; i < n; i++) {
          const b = i * 16
          posBuf[i * 3]     = view.getFloat32(b,      true)
          posBuf[i * 3 + 1] = view.getFloat32(b + 4,  true)
          posBuf[i * 3 + 2] = view.getFloat32(b + 8,  true)
          intBuf[i]         = view.getFloat32(b + 12, true)
        }
        frameRef.current = { positions: posBuf, intensities: intBuf, n }
        setMeta({ n, seq: env.seq, ts: env.ts })
        setClusters(env.clusters ?? [])
      } catch {}
    })
  }, [setOnMessage])

  const sendControl = (cmd) => send(JSON.stringify(cmd))

  return { connState, frameRef, meta, clusters, sendControl }
}
