import { useState, useEffect, useRef } from 'react'
import { useWebSocket } from './useWebSocket'
import { wsUrl, WS_PATHS } from '@/utils/wsConfig'

const MAX_PTS   = 131072
const HEADER_BYTES = 24
const posBuf    = new Float32Array(MAX_PTS * 3)
const intBuf    = new Float32Array(MAX_PTS)
const rawBuf    = new Uint8Array(MAX_PTS * 16)

export function useLidar() {
  const { connState, setOnMessage, send } = useWebSocket(wsUrl(WS_PATHS.lidar), { binaryType: 'arraybuffer' })
  const frameRef = useRef({ positions: posBuf, intensities: intBuf, n: 0 })
  const [meta, setMeta] = useState({ n: 0, seq: 0, ts: 0 })
  const [clusters, setClusters] = useState([])
  const [detections, setDetections] = useState([])
  const [detectionMeta, setDetectionMeta] = useState({ status: 'waiting for detector', source: '', elapsedMs: 0, ts: 0 })

  useEffect(() => {
    setOnMessage((e) => {
      try {
        if (e.data instanceof ArrayBuffer) {
          readBinaryFrame(e.data, frameRef, setMeta, setClusters)
          return
        }

        const env = JSON.parse(e.data)
        if (env.type === 'detections') {
          setDetections(Array.isArray(env.boxes) ? env.boxes : [])
          setDetectionMeta({
            status: env.status ?? '',
            source: env.source ?? env.mode ?? '',
            elapsedMs: env.elapsed_ms ?? 0,
            ts: env.ts ?? 0,
          })
          return
        }
        if (env.type !== 'frame') return

        readJsonFrame(env, frameRef, setMeta)
        setClusters(env.clusters ?? [])
      } catch {}
    })
  }, [setOnMessage])

  const sendControl = (cmd) => send(JSON.stringify(cmd))

  return { connState, frameRef, meta, clusters, detections, detectionMeta, sendControl }
}

function readBinaryFrame(buffer, frameRef, setMeta, setClusters) {
  if (buffer.byteLength < HEADER_BYTES) return

  const header = new DataView(buffer, 0, HEADER_BYTES)
  const magic =
    String.fromCharCode(header.getUint8(0)) +
    String.fromCharCode(header.getUint8(1)) +
    String.fromCharCode(header.getUint8(2)) +
    String.fromCharCode(header.getUint8(3))

  if (magic !== 'PCLD') return

  const seq = header.getUint32(8, true)
  const requested = header.getUint32(12, true)
  const ts = header.getFloat64(16, true)
  const available = Math.floor((buffer.byteLength - HEADER_BYTES) / 16)
  const n = Math.min(requested, available, MAX_PTS)
  const points = new Float32Array(buffer, HEADER_BYTES, n * 4)

  for (let i = 0; i < n; i++) {
    const src = i * 4
    const dst = i * 3
    posBuf[dst] = points[src]
    posBuf[dst + 1] = points[src + 1]
    posBuf[dst + 2] = points[src + 2]
    intBuf[i] = points[src + 3]
  }

  frameRef.current = { positions: posBuf, intensities: intBuf, n }
  setMeta({ n, seq, ts })
  setClusters([])
}

function readJsonFrame(env, frameRef, setMeta) {
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
}
