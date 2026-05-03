import { useState, useEffect } from 'react'
import { useWebSocket } from './useWebSocket'
import { wsUrl, WS_PATHS } from '@/utils/wsConfig'

export function useThreats() {
  const { connState, setOnMessage } = useWebSocket(wsUrl(WS_PATHS.threats))
  const [threats, setThreats] = useState([])
  const [stats, setStats] = useState({ total: 0, critical: 0, ts: 0 })

  useEffect(() => {
    setOnMessage((e) => {
      try {
        const env = JSON.parse(e.data)
        if (env.type !== 'frame') return
        const list = env.threats ?? []
        setThreats(list)
        setStats({
          total:    list.length,
          critical: list.filter(t => t.confidence >= 0.8).length,
          ts:       env.ts,
        })
      } catch {}
    })
  }, [setOnMessage])

  return { connState, threats, stats }
}
