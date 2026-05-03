import { useEffect, useRef, useCallback, useState } from 'react'

export function useWebSocket(url, options = {}) {
  const wsRef       = useRef(null)
  const retryDelay  = useRef(1000)
  const retryTimer  = useRef(null)
  const onMsgRef    = useRef(null)
  const [connState, setConnState] = useState('connecting')
  const binaryType = options.binaryType ?? 'blob'

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    setConnState('connecting')

    const sock = new WebSocket(url)
    sock.binaryType = binaryType
    wsRef.current = sock

    sock.onopen = () => {
      retryDelay.current = 1000
      setConnState('live')
    }
    sock.onclose = () => {
      setConnState('offline')
      retryTimer.current = setTimeout(() => {
        retryDelay.current = Math.min(retryDelay.current * 2, 16000)
        connect()
      }, retryDelay.current)
    }
    sock.onerror = () => setConnState('offline')  // browser logs the URL; no need to re-log here
    sock.onmessage = (e) => onMsgRef.current?.(e)
  }, [url, binaryType])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(retryTimer.current)
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.close()
      }
    }
  }, [connect])

  const setOnMessage = useCallback((fn) => { onMsgRef.current = fn }, [])

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(data)
  }, [])

  return { connState, setOnMessage, send }
}
