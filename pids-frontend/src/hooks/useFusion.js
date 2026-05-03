import { useState, useEffect } from 'react'
import { useWebSocket } from './useWebSocket'
import { wsUrl, WS_PATHS } from '@/utils/wsConfig'

const EMPTY_FRAME = {
  map: null,
  tracks: [],
  zones: [],
  pose: null,
  stats: { occupied: 0, mapped: 0 },
  seq: 0,
  ts: 0,
}

export function useFusion() {
  const { connState, setOnMessage, send } = useWebSocket(wsUrl(WS_PATHS.fusion))
  const [frame, setFrame] = useState(EMPTY_FRAME)

  useEffect(() => {
    setOnMessage((e) => {
      try {
        const env = JSON.parse(e.data)
        if (env.type !== 'frame') return

        const map = env.map ?? env.occupancy ?? null
        const tracks = env.tracks ?? env.threats ?? []
        const zones = env.zones ?? []
        const cells = Array.isArray(map?.cells) ? map.cells : null
        const occupied = cells
          ? cells.reduce((n, c) => n + (cellValue(c) > 0.55 ? 1 : 0), 0)
          : 0
        const mapped = cells?.length ?? 0

        setFrame({
          map,
          tracks,
          zones,
          pose: env.pose ?? null,
          stats: { occupied, mapped },
          seq: env.seq ?? 0,
          ts: env.ts ?? 0,
        })
      } catch {}
    })
  }, [setOnMessage])

  const sendControl = (cmd) => send(JSON.stringify(cmd))

  return { connState, frame, sendControl }
}

function cellValue(cell) {
  if (typeof cell === 'number') return cell
  return cell?.occupancy ?? cell?.occ ?? cell?.value ?? 0
}
