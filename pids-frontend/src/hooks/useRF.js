import { useState, useEffect, useRef } from 'react'
import { useWebSocket } from './useWebSocket'
import { wsUrl, WS_PATHS } from '@/utils/wsConfig'

const MAX_FFT = 4096
const fftBuf  = new Float32Array(MAX_FFT)
const rawBuf  = new Uint8Array(MAX_FFT * 4)

export const RF_BANDS = {
  'ISM 433':    { centerFreq: 433.92e6, sampleRate: 2.4e6 },
  'ISM 868':    { centerFreq: 868e6,    sampleRate: 2.4e6 },
  'ISM 915':    { centerFreq: 915e6,    sampleRate: 2.4e6 },
  'ADS-B 1090': { centerFreq: 1090e6,  sampleRate: 2.4e6 },
  'FM Radio':   { centerFreq: 96e6,    sampleRate: 10e6  },
  'GPS L1':     { centerFreq: 1575.42e6, sampleRate: 2.4e6 },
  'Cellular UL':{ centerFreq: 880e6,   sampleRate: 5e6   },
}

export function useRF() {
  const { connState, setOnMessage, send } = useWebSocket(wsUrl(WS_PATHS.rf))
  const fftRef = useRef(fftBuf)
  const [meta, setMeta] = useState({
    centerFreq: 433.92e6,
    sampleRate: 2.4e6,
    noiseFloor: -85,
    nFft: 0,
    ts: 0,
  })
  const [peaks, setPeaks] = useState([])

  useEffect(() => {
    setOnMessage((e) => {
      try {
        const env = JSON.parse(e.data)
        if (env.type !== 'frame') return

        const str = atob(env.data)
        const len = Math.min(str.length, rawBuf.length)
        for (let i = 0; i < len; i++) rawBuf[i] = str.charCodeAt(i)
        const view = new DataView(rawBuf.buffer, 0, len)
        const n = Math.min(env.n_fft ?? 1024, MAX_FFT)
        for (let i = 0; i < n; i++) fftBuf[i] = view.getFloat32(i * 4, true)
        fftRef.current = fftBuf

        setMeta({
          centerFreq: env.center_freq,
          sampleRate: env.sample_rate,
          noiseFloor: env.noise_floor ?? -85,
          nFft: n,
          ts: env.ts,
        })
        setPeaks(env.peaks ?? [])
      } catch {}
    })
  }, [setOnMessage])

  const setBand = (band) => {
    const cfg = RF_BANDS[band]
    if (cfg) send(JSON.stringify({ type: 'set_freq', ...cfg }))
  }

  const setFreq = (centerFreq, sampleRate) => {
    send(JSON.stringify({ type: 'set_freq', center_freq: centerFreq, sample_rate: sampleRate }))
  }

  return { connState, fftRef, meta, peaks, setBand, setFreq }
}
